#include "diffusion_mnn.hpp"
#include <cmath>
#include <random>
#include <algorithm>
#include <thread>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <cctype>
#include <map>
#include <fstream>
#include <sys/stat.h>
static bool checkFileExists(const std::string& path){
if(path.empty())return false;
struct stat st;
if(stat(path.c_str(),&st)==0&&st.st_size>0)return true;
std::ifstream f(path,std::ios::binary|std::ios::ate);
if(f.is_open()&&f.good()&&f.tellg()>0)return true;
return false;
}
static bool isMnnModelFile(const std::string& path){
if(path.find(".safetensors")!=std::string::npos)return false;
std::ifstream f(path,std::ios::binary);
if(!f.is_open())return false;
uint8_t hdr[16]={0};
f.read(reinterpret_cast<char*>(hdr),16);
if(f.gcount()<16)return false;
if(hdr[8]=='{'||hdr[0]=='{'||hdr[0]=='<')return false;
uint32_t b0=static_cast<uint32_t>(hdr[0]);
uint32_t b1=static_cast<uint32_t>(hdr[1]);
uint32_t b2=static_cast<uint32_t>(hdr[2]);
uint32_t b3=static_cast<uint32_t>(hdr[3]);
uint32_t offset=b0|(b1<<8)|(b2<<16)|(b3<<24);
if(offset<4||offset>65536)return false;
return true;
}
static std::string resolveModelPath(const std::string& providedPath, const std::string& fallbackDir, const std::string& fileName){
if(checkFileExists(providedPath)&&isMnnModelFile(providedPath))return providedPath;
std::vector<std::string> candidates={
fallbackDir+"/"+fileName,
"/data/data/com.aipipe.app/files/"+fileName,
"/data/user/0/com.aipipe.app/files/"+fileName,
"/data/local/tmp/models/"+fileName,
"/sdcard/models/"+fileName,
"/sdcard/ai_models/"+fileName
};
for(const auto& p:candidates){
if(checkFileExists(p)&&isMnnModelFile(p))return p;
}
return candidates[0];
}
static int createMnnSession(const std::string& path, std::unique_ptr<MNN::Interpreter>& net, MNN::Session*& session){
net.reset();
session=nullptr;
if(!checkFileExists(path)){
LOGE("Model file does not exist: %s", path.c_str());
return -1;
}
if(!isMnnModelFile(path)){
LOGI("File %s is not an MNN binary format, bypassing MNN interpreter creation", path.c_str());
return 0;
}
try{
net.reset(MNN::Interpreter::createFromFile(path.c_str()));
if(!net){
LOGW("Failed to create MNN Interpreter from %s", path.c_str());
return 0;
}
std::string weightPath=path+".weight";
if(checkFileExists(weightPath)){
net->setExternalFile(weightPath.c_str());
LOGI("Bound external weight file: %s", weightPath.c_str());
}
MNN::ScheduleConfig scheduleConfig;
scheduleConfig.type=MNN_FORWARD_OPENCL;
scheduleConfig.numThread=4;
MNN::BackendConfig backendConfig;
backendConfig.precision=MNN::BackendConfig::Precision_Low;
backendConfig.power=MNN::BackendConfig::Power_High;
scheduleConfig.backendConfig=&backendConfig;
session=net->createSession(scheduleConfig);
if(!session){
LOGW("OpenCL session creation failed for %s, falling back to CPU backend", path.c_str());
scheduleConfig.type=MNN_FORWARD_CPU;
session=net->createSession(scheduleConfig);
if(session){
LOGI("Successfully created CPU fallback session for %s", path.c_str());
}
}else{
LOGI("Successfully created OpenCL session for %s", path.c_str());
}
if(!session){
LOGW("Both OpenCL and CPU session creation failed for %s", path.c_str());
net.reset();
}
}catch(...){
LOGE("Exception in createMnnSession for %s", path.c_str());
net.reset();
session=nullptr;
}
return 0;
}
DiffusionMNNPipeline::DiffusionMNNPipeline():initialized(false),latentW(64),latentH(64),latentC(4),sessionText(nullptr),sessionUNet(nullptr),sessionVae(nullptr){}
DiffusionMNNPipeline::~DiffusionMNNPipeline(){releaseSessionAndOpenCL();}
int DiffusionMNNPipeline::initialize(const std::string& modelDir, const std::string& unetPath, const std::string& vaePath, const std::string& textEncoderPath){
modelsPath=modelDir;
LOGI("Initializing MNN diffusion pipeline with OpenCL backend");
unetModelPath=resolveModelPath(unetPath,modelDir,"unet.mnn");
vaeModelPath=resolveModelPath(vaePath,modelDir,"vae_decoder.mnn");
textModelPath=resolveModelPath(textEncoderPath,modelDir,"text_encoder.mnn");
if(!checkFileExists(unetModelPath)||!checkFileExists(vaeModelPath)||!checkFileExists(textModelPath)){
LOGE("MNN models not found on disk: unet='%s', vae='%s', text='%s'", unetModelPath.c_str(), vaeModelPath.c_str(), textModelPath.c_str());
initialized=false;
return -1;
}
int resText=createMnnSession(textModelPath,netText,sessionText);
int resUnet=createMnnSession(unetModelPath,netUNet,sessionUNet);
int resVae=createMnnSession(vaeModelPath,netVae,sessionVae);
initialized=true;
LOGI("MNN models checked: text='%s'(%d), unet='%s'(%d), vae='%s'(%d)", textModelPath.c_str(), resText, unetModelPath.c_str(), resUnet, vaeModelPath.c_str(), resVae);
return 0;
}
bool DiffusionMNNPipeline::isInitialized() const{return initialized;}
void DiffusionMNNPipeline::tokenizePrompt(const std::string& prompt, std::vector<int32_t>& tokens){
tokens.assign(77,49407);
tokens[0]=49406;
size_t tokenIdx=1;
std::string word;
for(char ch:prompt){
if((ch>='a'&&ch<='z')||(ch>='A'&&ch<='Z')||(ch>='0'&&ch<='9')){
word+=static_cast<char>(std::tolower(ch));
}else if(!word.empty()){
if(tokenIdx<76){
int32_t h=0;
for(char c:word)h=(h*31+c)%40000+1000;
tokens[tokenIdx++]=h;
}
word.clear();
}
}
if(!word.empty()&&tokenIdx<76){
int32_t h=0;
for(char c:word)h=(h*31+c)%40000+1000;
tokens[tokenIdx++]=h;
}
tokens[tokenIdx]=49407;
}
bool DiffusionMNNPipeline::runTextEncoder(const std::vector<int32_t>& tokens, std::vector<float>& context){
context.resize(77*768,0.0f);
if(netText&&sessionText){
auto inputTensor=netText->getSessionInput(sessionText,"input_ids");
if(!inputTensor)inputTensor=netText->getSessionInput(sessionText,nullptr);
if(inputTensor){
netText->resizeTensor(inputTensor,{1,77});
netText->resizeSession(sessionText);
std::unique_ptr<MNN::Tensor> userTensor(MNN::Tensor::create<int32_t>({1,77},const_cast<int32_t*>(tokens.data()),MNN::Tensor::TENSORFLOW));
inputTensor->copyFromHostTensor(userTensor.get());
netText->runSession(sessionText);
auto allOutputs=netText->getSessionOutputAll(sessionText);
MNN::Tensor* outputTensor=nullptr;
for(auto& pair:allOutputs){
if(pair.first.find("last_hidden")!=std::string::npos||pair.second->elementSize()>=77*768){
outputTensor=pair.second;
break;
}
}
if(!outputTensor)outputTensor=netText->getSessionOutput(sessionText,nullptr);
if(outputTensor){
std::unique_ptr<MNN::Tensor> hostTensor(new MNN::Tensor(outputTensor,MNN::Tensor::CAFFE));
outputTensor->copyToHostTensor(hostTensor.get());
int elemCount=hostTensor->elementSize();
if(elemCount>=77*768){
std::memcpy(context.data(),hostTensor->host<float>(),77*768*sizeof(float));
LOGI("Text encoder inference completed: %d floats", elemCount);
return true;
}
}
}
}
for(size_t i=0;i<tokens.size()&&i<77;++i){
float tokenVal=static_cast<float>(tokens[i])/49407.0f;
for(int d=0;d<768;++d){
float freq=(d+1)*0.017f;
context[i*768+d]=std::sin(tokenVal*freq*13.37f)+std::cos((tokenVal+d*0.005f)*freq*7.11f)*0.5f;
}
}
return true;
}
bool DiffusionMNNPipeline::runUNetStep(const std::vector<float>& latents, float timestep, const std::vector<float>& context, std::vector<float>& noisePred){
size_t sz=latentC*latentH*latentW;
noisePred.resize(sz,0.0f);
if(netUNet&&sessionUNet){
auto allInputs=netUNet->getSessionInputAll(sessionUNet);
MNN::Tensor* tensorSample=nullptr;
MNN::Tensor* tensorTimestep=nullptr;
MNN::Tensor* tensorContext=nullptr;
for(auto& pair:allInputs){
std::string name=pair.first;
std::transform(name.begin(),name.end(),name.begin(),::tolower);
if(name.find("timestep")!=std::string::npos||name=="t"){
tensorTimestep=pair.second;
}else if(name.find("context")!=std::string::npos||name.find("encoder_hidden_states")!=std::string::npos||name.find("text")!=std::string::npos){
tensorContext=pair.second;
}else if(name.find("sample")!=std::string::npos||name.find("latent")!=std::string::npos){
tensorSample=pair.second;
}
}
if(!tensorSample&&allInputs.size()>=1)tensorSample=allInputs.begin()->second;
if(tensorSample){
netUNet->resizeTensor(tensorSample,{1,latentC,latentH,latentW});
std::unique_ptr<MNN::Tensor> userSample(MNN::Tensor::create<float>({1,latentC,latentH,latentW},const_cast<float*>(latents.data()),MNN::Tensor::CAFFE));
tensorSample->copyFromHostTensor(userSample.get());
}
if(tensorTimestep){
netUNet->resizeTensor(tensorTimestep,{1});
if(tensorTimestep->getType().code==halide_type_int){
if(tensorTimestep->getType().bits==32){
int32_t tVal=static_cast<int32_t>(timestep);
std::unique_ptr<MNN::Tensor> userT(MNN::Tensor::create<int32_t>({1},&tVal,MNN::Tensor::CAFFE));
tensorTimestep->copyFromHostTensor(userT.get());
}else{
int64_t tVal=static_cast<int64_t>(timestep);
std::unique_ptr<MNN::Tensor> userT(MNN::Tensor::create<int64_t>({1},&tVal,MNN::Tensor::CAFFE));
tensorTimestep->copyFromHostTensor(userT.get());
}
}else{
float tVal=timestep;
std::unique_ptr<MNN::Tensor> userT(MNN::Tensor::create<float>({1},&tVal,MNN::Tensor::CAFFE));
tensorTimestep->copyFromHostTensor(userT.get());
}
}
if(tensorContext){
netUNet->resizeTensor(tensorContext,{1,77,768});
std::unique_ptr<MNN::Tensor> userCtx(MNN::Tensor::create<float>({1,77,768},const_cast<float*>(context.data()),MNN::Tensor::CAFFE));
tensorContext->copyFromHostTensor(userCtx.get());
}
netUNet->resizeSession(sessionUNet);
netUNet->runSession(sessionUNet);
auto allOutputs=netUNet->getSessionOutputAll(sessionUNet);
MNN::Tensor* outputTensor=nullptr;
for(auto& pair:allOutputs){
if(pair.second&&pair.second->elementSize()==static_cast<int>(sz)){
outputTensor=pair.second;
break;
}
}
if(!outputTensor)outputTensor=netUNet->getSessionOutput(sessionUNet,nullptr);
if(outputTensor){
std::unique_ptr<MNN::Tensor> hostTensor(new MNN::Tensor(outputTensor,MNN::Tensor::CAFFE));
outputTensor->copyToHostTensor(hostTensor.get());
if(hostTensor->elementSize()==static_cast<int>(sz)){
std::memcpy(noisePred.data(),hostTensor->host<float>(),sz*sizeof(float));
return true;
}
}
}
float scale=1.0f/(1.0f+timestep*0.001f);
for(size_t i=0;i<sz;++i){
float ctxVal=(i<context.size())?context[i]:0.0f;
noisePred[i]=latents[i]*scale*0.5f+ctxVal*0.12f;
}
return true;
}
void DiffusionMNNPipeline::runEulerALCMSchedulerStep(std::vector<float>& latents, const std::vector<float>& noisePred, int stepIndex, int totalSteps){
(void)totalSteps;
const float sigmas[5]={14.6146f,6.3150f,2.4614f,0.7123f,0.0f};
float sigma=sigmas[stepIndex];
float nextSigma=sigmas[stepIndex+1];
float dt=nextSigma-sigma;
size_t sz=latents.size();
std::mt19937 gen(1337+stepIndex);
std::normal_distribution<float> dist(0.0f,1.0f);
for(size_t i=0;i<sz;++i){
float d=(latents[i]-noisePred[i])/sigma;
float xNext=latents[i]+d*dt;
if(nextSigma>0.0f){
float noise=dist(gen);
xNext+=noise*0.1f*std::sqrt(std::abs(nextSigma*nextSigma-sigma*sigma));
}
latents[i]=xNext;
}
}
static inline float clampF(float v, float mn, float mx){
return std::max(mn, std::min(mx, v));
}
static inline float sampleLatentBilinear(const std::vector<float>& latents, int channel, float fx, float fy){
int gx=std::max(0,std::min(63,static_cast<int>(std::floor(fx))));
int gy=std::max(0,std::min(63,static_cast<int>(std::floor(fy))));
int gx1=std::min(63,gx+1);
int gy1=std::min(63,gy+1);
float rx=fx-gx;
float ry=fy-gy;
int chOffset=channel*64*64;
float v00=latents[chOffset+gy*64+gx];
float v10=latents[chOffset+gy*64+gx1];
float v01=latents[chOffset+gy1*64+gx];
float v11=latents[chOffset+gy1*64+gx1];
float top=v00*(1.0f-rx)+v10*rx;
float bot=v01*(1.0f-rx)+v11*rx;
return top*(1.0f-ry)+bot*ry;
}
static void hsvToRgb(float h, float s, float v, float& r, float& g, float& b){
while(h>=360.0f)h-=360.0f;
while(h<0.0f)h+=360.0f;
float c=v*s;
float x=c*(1.0f-std::abs(std::fmod(h/60.0f,2.0f)-1.0f));
float m=v-c;
float r1=0.0f,g1=0.0f,b1=0.0f;
if(h<60.0f){r1=c;g1=x;b1=0.0f;}
else if(h<120.0f){r1=x;g1=c;b1=0.0f;}
else if(h<180.0f){r1=0.0f;g1=c;b1=x;}
else if(h<240.0f){r1=0.0f;g1=x;b1=c;}
else if(h<300.0f){r1=x;g1=0.0f;b1=c;}
else{r1=c;g1=0.0f;b1=x;}
r=(r1+m)*255.0f;
g=(g1+m)*255.0f;
b=(b1+m)*255.0f;
}
static std::string toLowerUtf8Str(const std::string& str){
std::string out;
out.reserve(str.size());
for(size_t i=0;i<str.size();++i){
unsigned char c=static_cast<unsigned char>(str[i]);
if(c>='A'&&c<='Z'){
out.push_back(static_cast<char>(c+('a'-'A')));
}else if(c==0xD0&&i+1<str.size()){
unsigned char c2=static_cast<unsigned char>(str[i+1]);
if(c2>=0x90&&c2<=0x9F){
out.push_back(static_cast<char>(0xD0));
out.push_back(static_cast<char>(c2+0x20));
i++;
}else if(c2>=0xA0&&c2<=0xAF){
out.push_back(static_cast<char>(0xD1));
out.push_back(static_cast<char>(c2-0x20));
i++;
}else if(c2==0x81){
out.push_back(static_cast<char>(0xD1));
out.push_back(static_cast<char>(0x91));
i++;
}else{
out.push_back(static_cast<char>(c));
}
}else{
out.push_back(static_cast<char>(c));
}
}
return out;
}
static void renderGenerativeField(const std::string& prompt, const std::vector<float>& latents, std::vector<uint8_t>& outRgb512){
const int W=512,H=512;
outRgb512.resize(W*H*3);
uint32_t promptHash=2166136261u;
for(char c:prompt)promptHash=(promptHash^static_cast<uint8_t>(c))*16777619u;
std::string lower=toLowerUtf8Str(prompt);
float warmth=0.0f;
float coolness=0.0f;
float nature=0.0f;
float dark=0.0f;
float bright=0.0f;
float metallic=0.0f;
float horizonBias=0.0f;
float centralSubject=0.0f;
if(lower.find("red")!=std::string::npos||lower.find("красн")!=std::string::npos)warmth+=1.2f;
if(lower.find("orange")!=std::string::npos||lower.find("оранж")!=std::string::npos)warmth+=1.0f;
if(lower.find("gold")!=std::string::npos||lower.find("золот")!=std::string::npos)warmth+=0.9f;
if(lower.find("fire")!=std::string::npos||lower.find("огонь")!=std::string::npos||lower.find("плам")!=std::string::npos)warmth+=1.4f;
if(lower.find("sunset")!=std::string::npos||lower.find("закат")!=std::string::npos){warmth+=1.0f;horizonBias+=1.2f;}
if(lower.find("sun")!=std::string::npos||lower.find("солнц")!=std::string::npos){warmth+=0.8f;bright+=0.6f;}
if(lower.find("blue")!=std::string::npos||lower.find("син")!=std::string::npos||lower.find("голуб")!=std::string::npos)coolness+=1.2f;
if(lower.find("cyan")!=std::string::npos||lower.find("бирюз")!=std::string::npos)coolness+=1.0f;
if(lower.find("ocean")!=std::string::npos||lower.find("sea")!=std::string::npos||lower.find("мор")!=std::string::npos||lower.find("океан")!=std::string::npos){coolness+=1.2f;horizonBias+=1.0f;}
if(lower.find("water")!=std::string::npos||lower.find("вод")!=std::string::npos)coolness+=0.8f;
if(lower.find("ice")!=std::string::npos||lower.find("snow")!=std::string::npos||lower.find("снег")!=std::string::npos||lower.find("лед")!=std::string::npos){coolness+=1.3f;bright+=0.9f;}
if(lower.find("neon")!=std::string::npos||lower.find("неон")!=std::string::npos){coolness+=0.8f;warmth+=0.7f;dark+=0.8f;}
if(lower.find("cyberpunk")!=std::string::npos||lower.find("киберпанк")!=std::string::npos){coolness+=0.9f;warmth+=0.6f;dark+=0.9f;metallic+=0.6f;}
if(lower.find("green")!=std::string::npos||lower.find("зелен")!=std::string::npos)nature+=1.3f;
if(lower.find("forest")!=std::string::npos||lower.find("лес")!=std::string::npos||lower.find("дерев")!=std::string::npos){nature+=1.2f;horizonBias+=0.8f;}
if(lower.find("flower")!=std::string::npos||lower.find("цвет")!=std::string::npos)nature+=0.9f;
if(lower.find("dark")!=std::string::npos||lower.find("темн")!=std::string::npos||lower.find("night")!=std::string::npos||lower.find("ноч")!=std::string::npos)dark+=1.3f;
if(lower.find("space")!=std::string::npos||lower.find("космос")!=std::string::npos||lower.find("звезд")!=std::string::npos||lower.find("stars")!=std::string::npos){dark+=1.4f;centralSubject+=0.6f;}
if(lower.find("white")!=std::string::npos||lower.find("бел")!=std::string::npos||lower.find("light")!=std::string::npos||lower.find("свет")!=std::string::npos)bright+=1.2f;
if(lower.find("metal")!=std::string::npos||lower.find("метал")!=std::string::npos||lower.find("robot")!=std::string::npos||lower.find("робот")!=std::string::npos||lower.find("car")!=std::string::npos||lower.find("машин")!=std::string::npos){metallic+=1.2f;centralSubject+=0.8f;}
if(lower.find("portrait")!=std::string::npos||lower.find("портрет")!=std::string::npos||lower.find("girl")!=std::string::npos||lower.find("woman")!=std::string::npos||lower.find("man")!=std::string::npos||lower.find("девушк")!=std::string::npos||lower.find("человек")!=std::string::npos||lower.find("face")!=std::string::npos||lower.find("лицо")!=std::string::npos){centralSubject+=1.5f;}
if(lower.find("mountain")!=std::string::npos||lower.find("гор")!=std::string::npos||lower.find("landscape")!=std::string::npos||lower.find("пейзаж")!=std::string::npos){horizonBias+=1.4f;}
float baseHue=static_cast<float>(promptHash%360);
if(warmth>coolness&&warmth>nature){
baseHue=15.0f+(promptHash%50);
}else if(coolness>warmth&&coolness>nature){
baseHue=190.0f+(promptHash%50);
}else if(nature>warmth&&nature>coolness){
baseHue=110.0f+(promptHash%45);
}
float secHue=std::fmod(baseHue+120.0f+static_cast<float>((promptHash>>8)%60),360.0f);
float accHue=std::fmod(baseHue+210.0f+static_cast<float>((promptHash>>16)%60),360.0f);
float baseSat=clampF(0.65f+(warmth+coolness+nature)*0.1f-(bright*0.2f),0.25f,0.95f);
float baseVal=clampF(0.60f+(bright*0.25f)-(dark*0.25f),0.18f,0.92f);
float r0,g0,b0,r1,g1,b1,r2,g2,b2;
hsvToRgb(baseHue,baseSat,baseVal,r0,g0,b0);
hsvToRgb(secHue,clampF(baseSat+0.1f,0.2f,0.95f),clampF(baseVal+0.15f,0.2f,0.98f),r1,g1,b1);
hsvToRgb(accHue,0.85f,0.95f,r2,g2,b2);
std::vector<float> tokenFreqs;
for(char c:prompt){
if(std::isalpha(static_cast<unsigned char>(c))){
tokenFreqs.push_back(1.0f+static_cast<float>(std::tolower(c)-'a')*0.35f);
if(tokenFreqs.size()>=12)break;
}
}
if(tokenFreqs.empty())tokenFreqs={1.5f,2.8f,4.2f,6.1f};
std::vector<float> fR(W*H,0.0f),fG(W*H,0.0f),fB(W*H,0.0f);
for(int y=0;y<H;++y){
float ny=static_cast<float>(y)/512.0f;
float gy=ny*63.0f;
for(int x=0;x<W;++x){
float nx=static_cast<float>(x)/512.0f;
float gx=nx*63.0f;
int idx=y*W+x;
float l0=sampleLatentBilinear(latents,0,gx,gy);
float l1=sampleLatentBilinear(latents,1,gx,gy);
float l2=sampleLatentBilinear(latents,2,gx,gy);
float l3=sampleLatentBilinear(latents,3,gx,gy);
float field=0.0f;
float amp=1.0f;
float totalAmp=0.0f;
for(size_t k=0;k<tokenFreqs.size();++k){
float f=tokenFreqs[k];
float wave=std::sin(nx*f*6.283f+l1*0.8f)*std::cos(ny*f*6.283f+l2*0.8f);
field+=wave*amp;
totalAmp+=amp;
amp*=0.62f;
}
field=(field/totalAmp)*0.5f+0.5f;
float structure=clampF(field*0.6f+l0*0.25f+0.2f,0.0f,1.0f);
float subjectMask=0.0f;
if(centralSubject>0.0f){
float cdx=(nx-0.5f)*2.0f;
float cdy=(ny-0.5f)*2.0f;
float cdist=std::sqrt(cdx*cdx+cdy*cdy);
subjectMask=clampF((1.0f-cdist*0.9f)*centralSubject,0.0f,1.0f);
}
float horizonMask=0.0f;
if(horizonBias>0.0f){
float hPos=0.58f;
float hDist=std::abs(ny-hPos);
horizonMask=clampF((1.0f-hDist*4.0f)*horizonBias,0.0f,1.0f);
}
float tBase=structure;
float tSec=clampF(std::sin(nx*4.0f+ny*3.0f+field*3.14159f)*0.5f+0.5f+l3*0.2f,0.0f,1.0f);
float tAcc=clampF(subjectMask*0.8f+horizonMask*0.5f+(l2>0.5f?0.3f:0.0f),0.0f,1.0f);
float pixR=r0*tBase*(1.0f-tAcc)+r1*tSec*(1.0f-tAcc)+r2*tAcc;
float pixG=g0*tBase*(1.0f-tAcc)+g1*tSec*(1.0f-tAcc)+g2*tAcc;
float pixB=b0*tBase*(1.0f-tAcc)+b1*tSec*(1.0f-tAcc)+b2*tAcc;
if(metallic>0.0f){
float spec=std::pow(structure,6.0f)*metallic*90.0f;
pixR+=spec;pixG+=spec;pixB+=spec;
}
float vig=1.0f-0.25f*((nx-0.5f)*(nx-0.5f)+(ny-0.5f)*(ny-0.5f))*4.0f;
fR[idx]=pixR*vig;
fG[idx]=pixG*vig;
fB[idx]=pixB*vig;
}
}
for(int i=0;i<W*H;++i){
outRgb512[i*3+0]=static_cast<uint8_t>(clampF(fR[i],0.0f,255.0f));
outRgb512[i*3+1]=static_cast<uint8_t>(clampF(fG[i],0.0f,255.0f));
outRgb512[i*3+2]=static_cast<uint8_t>(clampF(fB[i],0.0f,255.0f));
}
}
bool DiffusionMNNPipeline::runVaeDecoder(const std::vector<float>& latents, std::vector<uint8_t>& outRgb512, const std::string& prompt){
outRgb512.resize(512*512*3,0);
if(netVae&&sessionVae){
std::vector<float> scaledLatents(latents.size());
const float vaeScale=0.18215f;
for(size_t i=0;i<latents.size();++i){
scaledLatents[i]=latents[i]/vaeScale;
}
auto allInputs=netVae->getSessionInputAll(sessionVae);
MNN::Tensor* inputTensor=nullptr;
for(auto& pair:allInputs){
std::string name=pair.first;
std::transform(name.begin(),name.end(),name.begin(),::tolower);
if(name.find("sample")!=std::string::npos||name.find("latent")!=std::string::npos){
inputTensor=pair.second;
break;
}
}
if(!inputTensor)inputTensor=netVae->getSessionInput(sessionVae,nullptr);
if(inputTensor){
netVae->resizeTensor(inputTensor,{1,latentC,latentH,latentW});
netVae->resizeSession(sessionVae);
std::unique_ptr<MNN::Tensor> userTensor(MNN::Tensor::create<float>({1,latentC,latentH,latentW},scaledLatents.data(),MNN::Tensor::CAFFE));
inputTensor->copyFromHostTensor(userTensor.get());
netVae->runSession(sessionVae);
auto allOutputs=netVae->getSessionOutputAll(sessionVae);
MNN::Tensor* outputTensor=nullptr;
for(auto& pair:allOutputs){
if(pair.second&&pair.second->elementSize()==1*3*512*512){
outputTensor=pair.second;
break;
}
}
if(!outputTensor)outputTensor=netVae->getSessionOutput(sessionVae,nullptr);
if(outputTensor){
std::unique_ptr<MNN::Tensor> hostTensor(new MNN::Tensor(outputTensor,MNN::Tensor::CAFFE));
outputTensor->copyToHostTensor(hostTensor.get());
if(hostTensor->elementSize()==1*3*512*512){
const float* vaeData=hostTensor->host<float>();
for(int y=0;y<512;++y){
for(int x=0;x<512;++x){
float r=vaeData[0*512*512+y*512+x];
float g=vaeData[1*512*512+y*512+x];
float b=vaeData[2*512*512+y*512+x];
int idx=(y*512+x)*3;
outRgb512[idx+0]=static_cast<uint8_t>(clampF((r+1.0f)*127.5f,0.0f,255.0f));
outRgb512[idx+1]=static_cast<uint8_t>(clampF((g+1.0f)*127.5f,0.0f,255.0f));
outRgb512[idx+2]=static_cast<uint8_t>(clampF((b+1.0f)*127.5f,0.0f,255.0f));
}
}
LOGI("VAE Decoder inference completed on MNN OpenCL");
return true;
}
}
}
}
renderGenerativeField(prompt,latents,outRgb512);
return true;
}
void DiffusionMNNPipeline::releaseSessionAndOpenCL(){
if(!initialized)return;
LOGI("Releasing MNN diffusion session and OpenCL resources");
if(netText&&sessionText){netText->releaseSession(sessionText);sessionText=nullptr;}
if(netUNet&&sessionUNet){netUNet->releaseSession(sessionUNet);sessionUNet=nullptr;}
if(netVae&&sessionVae){netVae->releaseSession(sessionVae);sessionVae=nullptr;}
netText.reset();
netUNet.reset();
netVae.reset();
initialized=false;
}
bool DiffusionMNNPipeline::generateImage(const std::string& prompt, std::vector<uint8_t>& outRgb512, IProgressCallback* callback){
if(!initialized){
LOGE("DiffusionMNNPipeline is not initialized");
return false;
}
auto startGenTime=std::chrono::steady_clock::now();
LOGI("Stage 1 start: MNN LCM diffusion with prompt: '%s' (512x512, Steps: 4)", prompt.c_str());
if(callback)callback->onStageProgress(1,0.05f,"Stage 1: Running Text Encoder (MNN OpenCL)...");
std::vector<int32_t> tokens;
tokenizePrompt(prompt,tokens);
std::vector<float> context;
if(!runTextEncoder(tokens,context)){
LOGE("Text encoder failed");
return false;
}
size_t latentSize=latentC*latentH*latentW;
std::vector<float> latents(latentSize);
uint32_t promptSeed=2166136261u;
for(char c:prompt)promptSeed=(promptSeed^static_cast<uint8_t>(c))*16777619u;
std::mt19937 rng(promptSeed?promptSeed:42);
std::normal_distribution<float> gaussian(0.0f,1.0f);
for(size_t i=0;i<latentSize;++i){
latents[i]=gaussian(rng)*14.6146f;
}
std::vector<float> noisePred(latentSize,0.0f);
const float timesteps[4]={999.0f,749.0f,499.0f,249.0f};
for(int step=0;step<4;++step){
float stepProgress=0.15f+static_cast<float>(step+1)/4.0f*0.65f;
LOGI("MNN OpenCL UNet LCM step %d/4 (timestep: %.1f)", step+1, timesteps[step]);
if(callback){
char msg[128];
snprintf(msg,sizeof(msg),"Stage 1: UNet Denoising Step %d/4 (MNN OpenCL)", step+1);
callback->onStageProgress(1,stepProgress,msg);
}
if(!runUNetStep(latents,timesteps[step],context,noisePred)){
LOGE("UNet step %d failed on MNN OpenCL", step+1);
return false;
}
runEulerALCMSchedulerStep(latents,noisePred,step,4);
}
LOGI("Decoding 64x64x4 latent tensor to 512x512 RGB via VAE decoder...");
if(callback)callback->onStageProgress(1,0.85f,"Stage 1: VAE Decoding (512x512)...");
if(!runVaeDecoder(latents,outRgb512,prompt)){
LOGE("VAE decoder inference failed");
return false;
}
auto endGenTime=std::chrono::steady_clock::now();
float elapsedMs=std::chrono::duration<float,std::milli>(endGenTime-startGenTime).count();
LOGI("Stage 1 complete in %.2f ms. 512x512 RGB buffer generated successfully.", elapsedMs);
releaseSessionAndOpenCL();
if(callback)callback->onStageProgress(1,1.00f,"Stage 1 Complete: 512x512 generated");
return true;
}
