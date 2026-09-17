#include "pipeline_common.hpp"
#include "diffusion_mnn.hpp"
#include "upscale_ncnn.hpp"
#include <memory>
#include <cstring>
#include <algorithm>
class JNIProgressBridge:public IProgressCallback{
public:
JNIProgressBridge(JNIEnv* env,jobject callbackObj):mEnv(env),mCallbackObj(callbackObj){
if(callbackObj){
jclass cls=env->GetObjectClass(callbackObj);
mMethodId=env->GetMethodID(cls,"onProgress","(IFLjava/lang/String;)V");
}else{
mMethodId=nullptr;
}
}
void onStageProgress(int stage,float progress,const char* statusMsg) override{
if(!mCallbackObj||!mMethodId)return;
jstring jMsg=mEnv->NewStringUTF(statusMsg?statusMsg:"");
mEnv->CallVoidMethod(mCallbackObj,mMethodId,static_cast<jint>(stage),static_cast<jfloat>(progress),jMsg);
mEnv->DeleteLocalRef(jMsg);
}
private:
JNIEnv* mEnv;
jobject mCallbackObj;
jmethodID mMethodId;
};
static bool writeBufferToAndroidBitmap(JNIEnv* env,jobject bitmap,const uint8_t* src,int width,int height,int channels){
if(!bitmap||!src)return false;
AndroidBitmapInfo info;
if(AndroidBitmap_getInfo(env,bitmap,&info)<0){
LOGE("Failed to get AndroidBitmap info");
return false;
}
if(info.format!=ANDROID_BITMAP_FORMAT_RGBA_8888){
LOGE("AndroidBitmap format is not RGBA_8888: %d", info.format);
return false;
}
void* pixels=nullptr;
if(AndroidBitmap_lockPixels(env,bitmap,&pixels)<0||!pixels){
LOGE("Failed to lock AndroidBitmap pixels");
return false;
}
for(int y=0;y<height&&y<static_cast<int>(info.height);++y){
uint8_t* dstRow=static_cast<uint8_t*>(pixels)+y*info.stride;
const uint8_t* srcRow=src+y*width*channels;
for(int x=0;x<width&&x<static_cast<int>(info.width);++x){
dstRow[x*4+0]=srcRow[x*channels+0];
dstRow[x*4+1]=srcRow[x*channels+1];
dstRow[x*4+2]=srcRow[x*channels+2];
dstRow[x*4+3]=0xFF;
}
}
AndroidBitmap_unlockPixels(env,bitmap);
return true;
}
static bool writeFloatBufferToAndroidBitmap(JNIEnv* env,jobject bitmap,const float* src,int width,int height,int channels,bool bipolar){
if(!bitmap||!src)return false;
AndroidBitmapInfo info;
if(AndroidBitmap_getInfo(env,bitmap,&info)<0)return false;
if(info.format!=ANDROID_BITMAP_FORMAT_RGBA_8888)return false;
void* pixels=nullptr;
if(AndroidBitmap_lockPixels(env,bitmap,&pixels)<0||!pixels)return false;
for(int y=0;y<height&&y<static_cast<int>(info.height);++y){
uint8_t* dstRow=static_cast<uint8_t*>(pixels)+y*info.stride;
const float* srcRow=src+y*width*channels;
for(int x=0;x<width&&x<static_cast<int>(info.width);++x){
float r=srcRow[x*channels+0];
float g=srcRow[x*channels+1];
float b=srcRow[x*channels+2];
float rf=bipolar?((r+1.0f)*127.5f):(r*255.0f);
float gf=bipolar?((g+1.0f)*127.5f):(g*255.0f);
float bf=bipolar?((b+1.0f)*127.5f):(b*255.0f);
dstRow[x*4+0]=static_cast<uint8_t>(std::clamp(rf,0.0f,255.0f));
dstRow[x*4+1]=static_cast<uint8_t>(std::clamp(gf,0.0f,255.0f));
dstRow[x*4+2]=static_cast<uint8_t>(std::clamp(bf,0.0f,255.0f));
dstRow[x*4+3]=0xFF;
}
}
AndroidBitmap_unlockPixels(env,bitmap);
return true;
}
static std::unique_ptr<DiffusionMNNPipeline> sDiffusionPipeline;
static std::unique_ptr<UpscaleNCNNPipeline> sUpscalePipeline;
extern "C" JNIEXPORT jboolean JNICALL
Java_com_aipipe_app_NativePipelineBridge_nativeInit(JNIEnv* env,jobject thiz,jstring modelDir_){
const char* modelDir=env->GetStringUTFChars(modelDir_,nullptr);
LOGI("Native init requested with modelDir: %s", modelDir);
sDiffusionPipeline=std::make_unique<DiffusionMNNPipeline>();
sUpscalePipeline=std::make_unique<UpscaleNCNNPipeline>();
bool mnnOk=sDiffusionPipeline->initialize(modelDir);
bool ncnnOk=sUpscalePipeline->initialize(modelDir);
env->ReleaseStringUTFChars(modelDir_,modelDir);
LOGI("Native engines initialized. MNN OpenCL: %s, NCNN Vulkan: %s", mnnOk?"OK":"FAIL", ncnnOk?"OK":"FAIL");
return static_cast<jboolean>(mnnOk&&ncnnOk);
}
extern "C" JNIEXPORT jobject JNICALL
Java_com_aipipe_app_NativePipelineBridge_nativeExecutePipeline(JNIEnv* env,jobject thiz,jstring prompt_,jint timeoutSec,jint targetW,jint targetH,jobject callback,jobject outBitmap512,jobject outBitmap4K){
auto pipelineStart=std::chrono::steady_clock::now();
const char* prompt=env->GetStringUTFChars(prompt_,nullptr);
LOGI("Starting execution pipeline. Timeout: %ds, Target: %dx%d, Prompt: '%s'", timeoutSec, targetW, targetH, prompt);
JNIProgressBridge progressBridge(env,callback);
if(!sDiffusionPipeline)sDiffusionPipeline=std::make_unique<DiffusionMNNPipeline>();
if(!sUpscalePipeline)sUpscalePipeline=std::make_unique<UpscaleNCNNPipeline>();
ExecutionMetrics metrics;
std::vector<uint8_t> rgba512;
auto diffStart=std::chrono::steady_clock::now();
bool diffOk=sDiffusionPipeline->generateImage(prompt,rgba512,&progressBridge);
auto diffEnd=std::chrono::steady_clock::now();
metrics.diffusionDurationMs=std::chrono::duration<float,std::milli>(diffEnd-diffStart).count();
env->ReleaseStringUTFChars(prompt_,prompt);
if(!diffOk){
LOGE("Stage 1 diffusion failed on Mali OpenCL backend");
return nullptr;
}
if(outBitmap512){
int ch=(rgba512.size()>=static_cast<size_t>(512*512*4))?4:3;
if(writeBufferToAndroidBitmap(env,outBitmap512,rgba512.data(),512,512,ch)){
LOGI("Locked and updated Stage 1 AndroidBitmap 512x512 with stride and ARGB_8888 0xFF alpha");
}
}
auto curTime=std::chrono::steady_clock::now();
float elapsedSecSoFar=std::chrono::duration<float>(curTime-pipelineStart).count();
float remainingBudgetSec=static_cast<float>(timeoutSec)-elapsedSecSoFar;
LOGI("Stage 1 completed in %.2f ms. Elapsed pipeline time: %.2f s. Remaining budget: %.2f s", metrics.diffusionDurationMs, elapsedSecSoFar, remainingBudgetSec);
std::vector<uint8_t> rgba4K;
bool usedFallback=false;
auto upStart=std::chrono::steady_clock::now();
bool upOk=sUpscalePipeline->upscaleImage(rgba512,512,512,rgba4K,targetW,targetH,remainingBudgetSec,usedFallback,&progressBridge);
auto upEnd=std::chrono::steady_clock::now();
metrics.upscaleDurationMs=std::chrono::duration<float,std::milli>(upEnd-upStart).count();
metrics.fallbackTriggered=usedFallback;
if(!upOk){
LOGE("Stage 2 upscale failed on Mali Vulkan backend");
return nullptr;
}
if(outBitmap4K){
int ch=(rgba4K.size()>=static_cast<size_t>(targetW*targetH*4))?4:3;
if(writeBufferToAndroidBitmap(env,outBitmap4K,rgba4K.data(),targetW,targetH,ch)){
LOGI("Locked and updated Stage 2 AndroidBitmap %dx%d with stride and ARGB_8888 0xFF alpha", targetW, targetH);
}
}
auto pipelineEnd=std::chrono::steady_clock::now();
metrics.totalDurationMs=std::chrono::duration<float,std::milli>(pipelineEnd-pipelineStart).count();
LOGI("Pipeline execution completed in %.2f ms (Budget: %d s, Budget Kept: %s). Fallback: %s", metrics.totalDurationMs, timeoutSec, (metrics.totalDurationMs<=timeoutSec*1000.0f)?"YES":"OVERDUE", usedFallback?"YES":"NO");
jclass resultClass=env->FindClass("com/aipipe/app/PipelineResult");
jmethodID constructor=env->GetMethodID(resultClass,"<init>","(FFFZZ)V");
jobject resultObj=env->NewObject(resultClass,constructor,metrics.diffusionDurationMs,metrics.upscaleDurationMs,metrics.totalDurationMs,static_cast<jboolean>(metrics.fallbackTriggered),static_cast<jboolean>(metrics.totalDurationMs<=timeoutSec*1000.0f));
return resultObj;
}
extern "C" JNIEXPORT void JNICALL
Java_com_aipipe_app_NativePipelineBridge_nativeRelease(JNIEnv* env,jobject thiz){
LOGI("Releasing native pipeline resources (MNN OpenCL + NCNN Vulkan)");
if(sDiffusionPipeline){
sDiffusionPipeline->releaseSessionAndOpenCL();
sDiffusionPipeline.reset();
}
if(sUpscalePipeline){
sUpscalePipeline->releaseVulkanInstance();
sUpscalePipeline.reset();
}
}
