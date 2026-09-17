#include "image_processing.hpp"
#include <cmath>
#include <algorithm>
#include <vector>
static inline float cubicWeight(float x){
const float a=-0.5f;
x=std::abs(x);
if(x<=1.0f)return(a+2.0f)*x*x*x-(a+3.0f)*x*x+1.0f;
if(x<2.0f)return a*x*x*x-5.0f*a*x*x+8.0f*a*x-4.0f*a;
return 0.0f;
}
static inline uint8_t clampPixel(float v){
if(v<0.0f)return 0;
if(v>255.0f)return 255;
return static_cast<uint8_t>(v+0.5f);
}
void bicubicUpscaleRGBA(const uint8_t* src, int srcW, int srcH, uint8_t* dst, int dstW, int dstH){
const float scaleX=static_cast<float>(srcW)/static_cast<float>(dstW);
const float scaleY=static_cast<float>(srcH)/static_cast<float>(dstH);
for(int dy=0;dy<dstH;++dy){
float sy=(dy+0.5f)*scaleY-0.5f;
int iy=static_cast<int>(std::floor(sy));
float fy=sy-iy;
float wy[4]={cubicWeight(fy+1.0f),cubicWeight(fy),cubicWeight(1.0f-fy),cubicWeight(2.0f-fy)};
uint8_t* dstRow=dst+dy*dstW*4;
for(int dx=0;dx<dstW;++dx){
float sx=(dx+0.5f)*scaleX-0.5f;
int ix=static_cast<int>(std::floor(sx));
float fx=sx-ix;
float wx[4]={cubicWeight(fx+1.0f),cubicWeight(fx),cubicWeight(1.0f-fx),cubicWeight(2.0f-fx)};
float r=0.0f,g=0.0f,b=0.0f,a=0.0f;
for(int m=0;m<4;++m){
int cy=std::clamp(iy-1+m,0,srcH-1);
const uint8_t* srcRow=src+cy*srcW*4;
float weightY=wy[m];
for(int n=0;n<4;++n){
int cx=std::clamp(ix-1+n,0,srcW-1);
float w=weightY*wx[n];
const uint8_t* p=srcRow+cx*4;
r+=p[0]*w;
g+=p[1]*w;
b+=p[2]*w;
a+=p[3]*w;
}
}
int dIdx=dx*4;
dstRow[dIdx+0]=clampPixel(r);
dstRow[dIdx+1]=clampPixel(g);
dstRow[dIdx+2]=clampPixel(b);
dstRow[dIdx+3]=clampPixel(a);
}
}
}
static inline float sinc(float x){
if(std::abs(x)<1e-5f)return 1.0f;
x*=3.14159265358979323846f;
return std::sin(x)/x;
}
static inline float lanczosWeight(float x){
x=std::abs(x);
if(x<2.0f)return sinc(x)*sinc(x*0.5f);
return 0.0f;
}
void lanczosUpscaleRGBA(const uint8_t* src, int srcW, int srcH, uint8_t* dst, int dstW, int dstH){
const float scaleX=static_cast<float>(srcW)/static_cast<float>(dstW);
const float scaleY=static_cast<float>(srcH)/static_cast<float>(dstH);
for(int dy=0;dy<dstH;++dy){
float sy=(dy+0.5f)*scaleY-0.5f;
int iy=static_cast<int>(std::floor(sy));
float fy=sy-iy;
float wy[4]={lanczosWeight(fy+1.0f),lanczosWeight(fy),lanczosWeight(1.0f-fy),lanczosWeight(2.0f-fy)};
float sumWy=wy[0]+wy[1]+wy[2]+wy[3];
if(sumWy>1e-5f){wy[0]/=sumWy;wy[1]/=sumWy;wy[2]/=sumWy;wy[3]/=sumWy;}
uint8_t* dstRow=dst+dy*dstW*4;
for(int dx=0;dx<dstW;++dx){
float sx=(dx+0.5f)*scaleX-0.5f;
int ix=static_cast<int>(std::floor(sx));
float fx=sx-ix;
float wx[4]={lanczosWeight(fx+1.0f),lanczosWeight(fx),lanczosWeight(1.0f-fx),lanczosWeight(2.0f-fx)};
float sumWx=wx[0]+wx[1]+wx[2]+wx[3];
if(sumWx>1e-5f){wx[0]/=sumWx;wx[1]/=sumWx;wx[2]/=sumWx;wx[3]/=sumWx;}
float r=0.0f,g=0.0f,b=0.0f,a=0.0f;
for(int m=0;m<4;++m){
int cy=std::clamp(iy-1+m,0,srcH-1);
const uint8_t* srcRow=src+cy*srcW*4;
float weightY=wy[m];
for(int n=0;n<4;++n){
int cx=std::clamp(ix-1+n,0,srcW-1);
float w=weightY*wx[n];
const uint8_t* p=srcRow+cx*4;
r+=p[0]*w;
g+=p[1]*w;
b+=p[2]*w;
a+=p[3]*w;
}
}
int dIdx=dx*4;
dstRow[dIdx+0]=clampPixel(r);
dstRow[dIdx+1]=clampPixel(g);
dstRow[dIdx+2]=clampPixel(b);
dstRow[dIdx+3]=clampPixel(a);
}
}
}
