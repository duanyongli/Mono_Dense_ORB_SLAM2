//
// Created by duan on 18-4-17.
//
#include <iostream>
#include <DepthPrediction.h>

namespace ORB_SLAM2
{
    DepthPrediction::DepthPrediction(std::string modelpath, int inchannels)
    :mstrmodelpath(modelpath), miInchannels(inchannels),mpFunPredictRGB(nullptr), mpFunPredictRGBD(nullptr)
    {
        initPython();
    }

    int DepthPrediction::initPython()
    {
        Py_Initialize();
        init_numpy();

        PyRun_SimpleString("import sys");
        PyRun_SimpleString(("sys.path.append('"+mstrmodelpath+"')").c_str());

        mpModule = PyImport_ImportModule("predict");
        if (!mpModule) {
            printf("Cant open python file!/n");
            return -1;
        }

        // 模块的字典列表
        mpDict = PyModule_GetDict(mpModule);
        if (!mpDict) {
            printf("Cant find dictionary./n");
            return -1;
        }

        PyObject *pArgs, *pValue;
        pArgs = PyTuple_New(1);
        assert(miInchannels==3 || miInchannels==4);
        pValue = PyLong_FromLong(miInchannels);
        PyTuple_SetItem(pArgs, 0, pValue);
        PyObject *ploadmodel = PyDict_GetItemString(mpDict, "load_model");
        PyObject_CallObject(ploadmodel, pArgs);
        Py_DECREF(pValue);
        Py_DECREF(pArgs);
        Py_DECREF(ploadmodel);


        mpFunPredictRGB = PyDict_GetItemString(mpDict, "predictRGB");
        mpFunPredictRGBD = PyDict_GetItemString(mpDict, "predictRGBD");

        return 0;
    }

    cv::Mat DepthPrediction::predictRGB(cv::Mat img)
    {
        int m, n;
        n = img.cols *3;
        m = img.rows;
        unsigned char *data = (unsigned char*)malloc(sizeof(unsigned char) * m * n);

        int p = 0;
        for (int i = 0; i < m;i++)
        {
            for (int j = 0; j < n; j++)
            {
                data[p]= img.at<unsigned char>(i, j);
                p++;
            }
        }

        npy_intp Dims[2]= { m, n }; //给定维度信息
        PyObject* PyArray = PyArray_SimpleNewFromData(2, Dims, NPY_UBYTE, data);
        PyObject* ArgArray = PyTuple_New(1);
        PyTuple_SetItem(ArgArray,0, PyArray);
        std::cout<<"start PyObject_CallObject"<<std::endl;
        PyArrayObject *pReturn = (PyArrayObject *)PyObject_CallObject(mpFunPredictRGB, ArgArray);
        std::cout<<"end PyObject_CallObject"<<std::endl;

        cv::Mat depth(480, 640, CV_32FC1, cv::Scalar(0.));
        int Rows = pReturn->dimensions[0], columns = pReturn->dimensions[1];
        for(int index_m = 0; index_m < Rows; ++index_m)
        {
            for(int index_n = 0; index_n < columns; ++index_n)
            {
                //访问数据，Index_m 和 Index_n 分别是数组元素的坐标，乘上相应维度的步长，即可以访问数组元素
                float val = *(float *)(pReturn->data + index_m * pReturn->strides[0] + index_n * pReturn->strides[1]);
                depth.at<float>(index_m, index_n) = val;
                //cout<<val<<endl;
            }
        }

        Py_DECREF(pReturn);

        return depth;
    }

    cv::Mat DepthPrediction::predictRGBD(cv::Mat img, cv::Mat sparseDepth)
    {
        int m, n;
        n = img.cols *3;
        m = img.rows;
        unsigned char *imgdata = (unsigned char*)malloc(sizeof(unsigned char) * m * n);

        int p = 0;
        for (int i = 0; i < m;i++)
        {
            for (int j = 0; j < n; j++)
            {
                imgdata[p]= img.at<unsigned char>(i, j);
                p++;
            }
        }
        npy_intp Dims[2]= { m, n }; //给定维度信息
        PyObject* PyimgArray = PyArray_SimpleNewFromData(2, Dims, NPY_UBYTE, imgdata);

        n = sparseDepth.cols;
        m = sparseDepth.rows;
        float* depthdata = (float*)malloc(sizeof(float) * m * n);
        p = 0;
        for (int i = 0; i < m;i++)
        {
            for (int j = 0; j < n; j++)
            {
                depthdata[p]= sparseDepth.at<float>(i, j);
                p++;
            }
        }
        npy_intp depthDims[2]= { m, n }; //给定维度信息
        PyObject* PydepthArray = PyArray_SimpleNewFromData(2, depthDims, NPY_FLOAT, depthdata);

        PyObject* ArgArray = PyTuple_New(2);
        PyTuple_SetItem(ArgArray,0, PyimgArray);
        PyTuple_SetItem(ArgArray,1, PydepthArray);
        std::cout<<"start PyObject_CallObject"<<std::endl;
        PyArrayObject *pReturn = (PyArrayObject *)PyObject_CallObject(mpFunPredictRGBD, ArgArray);
        std::cout<<"end PyObject_CallObject"<<std::endl;

        cv::Mat depth = cv::Mat::zeros(480, 640, CV_32F);
        int Rows = pReturn->dimensions[0], columns = pReturn->dimensions[1];
        for(int index_m = 0; index_m < Rows; ++index_m)
        {
            for(int index_n = 0; index_n < columns; ++index_n)
            {
                //访问数据，Index_m 和 Index_n 分别是数组元素的坐标，乘上相应维度的步长，即可以访问数组元素
                float val = *(float *)(pReturn->data + index_m * pReturn->strides[0] + index_n * pReturn->strides[1]);
                depth.at<float>(index_m, index_n) = val;
                //cout<<val<<endl;
            }
        }

        Py_DECREF(pReturn);

        return depth;
    }

    DepthPrediction::~DepthPrediction()
    {
        if(mpFunPredictRGB)
            Py_DECREF(mpFunPredictRGB);
        if(mpFunPredictRGBD)
            Py_DECREF(mpFunPredictRGBD);
        Py_DECREF(mpDict);
        Py_DECREF(mpModule);

        Py_Finalize();
    }


}