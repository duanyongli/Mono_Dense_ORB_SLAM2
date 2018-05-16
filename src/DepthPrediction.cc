//
// Created by duan on 18-4-17.
//

#include <DepthPrediction.h>

namespace ORB_SLAM2
{
    DepthPrediction::DepthPrediction(std::string modelpath)
    :mstrmodelpath(modelpath)
    {}

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

        mpFunPredict = PyDict_GetItemString(mpDict, "predict");

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

        PyArrayObject *pReturn = (PyArrayObject *)PyObject_CallObject(mpFunPredict, ArgArray);

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

    DepthPrediction::~DepthPrediction()
    {
        Py_DECREF(mpFunPredict);
        Py_DECREF(mpDict);
        Py_DECREF(mpModule);

        Py_Finalize();
    }


}