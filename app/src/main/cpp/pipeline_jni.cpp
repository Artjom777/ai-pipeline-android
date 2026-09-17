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
extern "C" JNIEXPORT jint JNICALL
Java_com_aipipe_app_NativePipelineBridge_nativeInit(JNIEnv* env,jobject thiz,jstring modelDir_,jstring unetPath_,jstring vaePath_,jstring textEncoderPath_){
const char* modelDirC=modelDir_?env->GetStringUTFChars(modelDir_,nullptr):nullptr;
const char* unetPathC=unetPath_?env->GetStringUTFChars(unetPath_,nullptr):nullptr;
const char* vaePathC=vaePath_?env->GetStringUTFChars(vaePath_,nullptr):nullptr;
const char* textEncoderPathC=textEncoderPath_?env->GetStringUTFChars(textEncoderPath_,nullptr):nullptr;
std::string modelDir=modelDirC?modelDirC:"";
std::string unetPath=unetPathC?unetPathC:"";
std::string vaePath=vaePathC?vaePathC:"";
std::string textEncoderPath=textEncoderPathC?textEncoderPathC:"";
if(modelDir_&&modelDirC)env->ReleaseStringUTFChars(modelDir_,modelDirC);
if(unetPath_&&unetPathC)env->ReleaseStringUTFChars(unetPath_,unetPathC);
if(vaePath_&&vaePathC)env->ReleaseStringUTFChars(vaePath_,vaePathC);
if(textEncoderPath_&&textEncoderPathC)env->ReleaseStringUTFChars(textEncoderPath_,textEncoderPathC);
LOGI("Native init: modelDir='%s', unet='%s', vae='%s', text='%s'", modelDir.c_str(), unetPath.c_str(), vaePath.c_str(), textEncoderPath.c_str());
sDiffusionPipeline=std::make_unique<DiffusionMNNPipeline>();
sUpscalePipeline=std::make_unique<UpscaleNCNNPipeline>();
int mnnRes=sDiffusionPipeline->initialize(modelDir,unetPath,vaePath,textEncoderPath);
if(mnnRes!=0){
LOGE("Diffusion pipeline init failed with code %d", mnnRes);
return static_cast<jint>(mnnRes);
}
bool ncnnOk=sUpscalePipeline->initialize(modelDir);
if(!ncnnOk){
LOGW("Upscale pipeline init returned false");
}
return 0;
}
extern "C" JNIEXPORT jint JNICALL
Java_com_aipipe_app_NativePipelineBridge_nativeInitInternal(JNIEnv* env,jobject thiz,jstring modelDir_,jstring unetPath_,jstring vaePath_,jstring textEncoderPath_){
return Java_com_aipipe_app_NativePipelineBridge_nativeInit(env,thiz,modelDir_,unetPath_,vaePath_,textEncoderPath_);
}
extern "C" JNIEXPORT jobject JNICALL
Java_com_aipipe_app_NativePipelineBridge_nativeExecutePipeline(JNIEnv* env,jobject thiz,jstring prompt_,jint timeoutSec,jint targetW,jint targetH,jobject callback,jobject outBitmap512,jobject outBitmap4K){
auto pipelineStart=std::chrono::steady_clock::now();
const char* promptC=prompt_?env->GetStringUTFChars(prompt_,nullptr):nullptr;
std::string prompt=promptC?promptC:"";
LOGI("Starting execution pipeline. Timeout: %ds, Target: %dx%d, Prompt: '%s'", timeoutSec, targetW, targetH, prompt.c_str());
if(prompt_&&promptC)env->ReleaseStringUTFChars(prompt_,promptC);
JNIProgressBridge progressBridge(env,callback);
if(!sDiffusionPipeline||!sDiffusionPipeline->isInitialized()){
LOGE("Diffusion pipeline is not initialized before execution");
return nullptr;
}
if(!sUpscalePipeline)sUpscalePipeline=std::make_unique<UpscaleNCNNPipeline>();
ExecutionMetrics metrics;
std::vector<uint8_t> rgb512;
auto diffStart=std::chrono::steady_clock::now();
bool diffOk=sDiffusionPipeline->generateImage(prompt,rgb512,&progressBridge);
auto diffEnd=std::chrono::steady_clock::now();
metrics.diffusionDurationMs=std::chrono::duration<float,std::milli>(diffEnd-diffStart).count();
if(!diffOk){
LOGE("Stage 1 diffusion failed on Mali OpenCL backend");
return nullptr;
}
if(outBitmap512){
int ch=(rgb512.size()>=static_cast<size_t>(512*512*4))?4:3;
if(writeBufferToAndroidBitmap(env,outBitmap512,rgb512.data(),512,512,ch)){
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
bool upOk=sUpscalePipeline->upscaleImage(rgb512,512,512,rgba4K,targetW,targetH,remainingBudgetSec,usedFallback,&progressBridge);
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
if(!resultClass)return nullptr;
jmethodID constructor=env->GetMethodID(resultClass,"<init>","(FFFZZ)V");
if(!constructor)return nullptr;
jobject resultObj=env->NewObject(resultClass,constructor,metrics.diffusionDurationMs,metrics.upscaleDurationMs,metrics.totalDurationMs,static_cast<jboolean>(metrics.fallbackTriggered),static_cast<jboolean>(metrics.totalDurationMs<=timeoutSec*1000.0f));
return resultObj;
}
extern "C" JNIEXPORT jobject JNICALL
Java_com_aipipe_app_NativePipelineBridge_nativeExecutePipelineInternal(JNIEnv* env,jobject thiz,jstring prompt_,jint timeoutSec,jint targetW,jint targetH,jobject callback,jobject outBitmap512,jobject outBitmap4K){
return Java_com_aipipe_app_NativePipelineBridge_nativeExecutePipeline(env,thiz,prompt_,timeoutSec,targetW,targetH,callback,outBitmap512,outBitmap4K);
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
extern "C" JNIEXPORT void JNICALL
Java_com_aipipe_app_NativePipelineBridge_nativeReleaseInternal(JNIEnv* env,jobject thiz){
Java_com_aipipe_app_NativePipelineBridge_nativeRelease(env,thiz);
}
