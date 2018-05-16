//
// Created by duan on 18-4-17.
//

#ifndef ORB_SLAM2_DEPTHPREDICTION_H
#define ORB_SLAM2_DEPTHPREDICTION_H

#include <string>
#include <opencv2/highgui/highgui.hpp>
#include </usr/include/python3.5m/Python.h>
#include </usr/include/numpy/arrayobject.h>

namespace ORB_SLAM2
{

    class DepthPrediction
    {
    public:
        DepthPrediction(std::string modelpath);
        int initPython();

        cv::Mat predictRGB(cv::Mat imgRGB);
        cv::Mat predictRGBD(cv::Mat imgRGB, cv::Mat sparseDepth);

        int init_numpy(){
            //初始化 numpy 执行环境，主要是导入包
            import_array();
        }

        ~DepthPrediction();
    private:
        PyObject *mpModule;
        PyObject *mpDict;
        PyObject *mpFunPredict;

        std::string mstrmodelpath;
    };

}
#endif //ORB_SLAM2_DEPTHPREDICTION_H
