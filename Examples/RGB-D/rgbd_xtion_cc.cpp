/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <OpenNI.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <System.h>

using namespace std;
using namespace openni;
using namespace cv;

void showdevice(){
    // 获取设备信息  
    Array<DeviceInfo> aDeviceList;
    OpenNI::enumerateDevices(&aDeviceList);

    cout << "电脑上连接着 " << aDeviceList.getSize() << " 个体感设备." << endl;

    for (int i = 0; i < aDeviceList.getSize(); ++i)
    {
        cout << "设备 " << i << endl;
        const DeviceInfo& rDevInfo = aDeviceList[i];
        cout << "设备名： " << rDevInfo.getName() << endl;
        cout << "设备Id： " << rDevInfo.getUsbProductId() << endl;
        cout << "供应商名： " << rDevInfo.getVendor() << endl;
        cout << "供应商Id: " << rDevInfo.getUsbVendorId() << endl;
        cout << "设备URI: " << rDevInfo.getUri() << endl;

    }
}

Status initstream(Status& rc, Device& xtion, VideoStream& streamDepth, VideoStream& streamColor)
{
    rc = STATUS_OK;

    // 创建深度数据流
    rc = streamDepth.create(xtion, SENSOR_DEPTH);
    if (rc == STATUS_OK)
    {
        // 设置深度图像视频模式
        VideoMode mModeDepth;
        // 分辨率大小
        mModeDepth.setResolution(640, 480);
        // 每秒30帧
        mModeDepth.setFps(30);
        // 像素格式
        mModeDepth.setPixelFormat(PIXEL_FORMAT_DEPTH_1_MM);

        streamDepth.setVideoMode(mModeDepth);
        streamDepth.setMirroringEnabled(false);

        // 打开深度数据流
        rc = streamDepth.start();
        if (rc != STATUS_OK)
        {
            cerr << "无法打开深度数据流：" << OpenNI::getExtendedError() << endl;
            streamDepth.destroy();
        }
    }
    else
    {
        cerr << "无法创建深度数据流：" << OpenNI::getExtendedError() << endl;
    }

    // 创建彩色图像数据流
    rc = streamColor.create(xtion, SENSOR_COLOR);
    if (rc == STATUS_OK)
    {
        // 同样的设置彩色图像视频模式
        VideoMode mModeColor;
        mModeColor.setResolution(640, 480);
        mModeColor.setFps(30);
        mModeColor.setPixelFormat(PIXEL_FORMAT_RGB888);

        streamColor.setVideoMode(mModeColor);
                             streamColor.setMirroringEnabled(false);
        // 打开彩色图像数据流
        rc = streamColor.start();
        if (rc != STATUS_OK)
        {
            cerr << "无法打开彩色图像数据流：" << OpenNI::getExtendedError() << endl;
            streamColor.destroy();
        }
    }
    else
    {
        cerr << "无法创建彩色图像数据流：" << OpenNI::getExtendedError() << endl;
    }

    if (!streamColor.isValid() || !streamDepth.isValid())
    {
        cerr << "彩色或深度数据流不合法" << endl;
        OpenNI::shutdown();
        rc = STATUS_ERROR;
        return rc;
    }

    // 图像模式注册,彩色图与深度图对齐
    if (xtion.isImageRegistrationModeSupported(
        IMAGE_REGISTRATION_DEPTH_TO_COLOR))
    {
        xtion.setImageRegistrationMode(IMAGE_REGISTRATION_DEPTH_TO_COLOR);
    }

    return rc;
}

int main(int argc, char **argv)
{
    if(argc != 3)
    {
        cerr << endl << "Usage: ./rgbd_cc path_to_vocabulary path_to_settings" << endl;
        return 1;
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::RGBD,true);

    cout << endl << "-------" << endl;
    cout << "Openning Xtion ..." << endl;

    Status rc = STATUS_OK;
    // 初始化OpenNI环境
    OpenNI::initialize();
    showdevice();
    // 声明并打开Device设备。
    Device xtion;
    const char * deviceURL = openni::ANY_DEVICE;  //设备名
    rc = xtion.open(deviceURL);
    
    VideoStream streamDepth;
    VideoStream streamColor;
    if(initstream(rc, xtion, streamDepth, streamColor) == STATUS_OK)
        cout << "Open Xtion Successfully!"<<endl;
    else
    {
        cout << "Open Xtion Failed!"<<endl;
        return 0;
    }

    // Main loop
    cv::Mat imRGB, imD;
    bool continueornot = true;
    // 循环读取数据流信息并保存在VideoFrameRef中
    VideoFrameRef  frameDepth;
    VideoFrameRef  frameColor;
    namedWindow("RGB Image", CV_WINDOW_AUTOSIZE);
    for (double index = 1.0; continueornot; index+=1.0)
    {
        rc = streamDepth.readFrame(&frameDepth);
        if (rc == STATUS_OK)
        {
            imD = cv::Mat(frameDepth.getHeight(), frameDepth.getWidth(), CV_16UC1, (void*)frameDepth.getData());
        }
        rc = streamColor.readFrame(&frameColor);
        if (rc == STATUS_OK)
        {
            const Mat tImageRGB(frameColor.getHeight(), frameColor.getWidth(), CV_8UC3, (void*)frameColor.getData());
 //           tImageRGB.copyTo(imRGB);
            
            cvtColor(tImageRGB, imRGB, CV_RGB2BGR);
            imshow("RGB Image",imRGB);
        }
        SLAM.TrackRGBD( imRGB, imD,  index);
        char c  = cv::waitKey(5);
        switch(c)
        {
        case 'q':
        case 27:
            continueornot = false;
            break;
        case 'p':
            cv::waitKey(0);
            break;
        default:
            break;
        }
    }
    // Stop all threads
    SLAM.Shutdown();
    SLAM.SaveTrajectoryTUM("trajectory.txt");
    cv::destroyAllWindows();
    return 0;
}

