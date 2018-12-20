from ctypes import *

dll=WinDLL("libFPDev_zz.dll")

#1:
# /** 功能: 获取设备状态
#     参数: nPortNo-端口号(0-USB, 1-COM1, 2-COM2, ..., 9-COM9)
# 	 返回: 0-在线, 其它-离线
# */
func1=dll.FPIDevDetect(0)
print("func1:",func1)

#2:
# 功能：采集图像并保存
# 参数：fileName：open（）函数文件名
def getAndSaveImg(fileName):
    TzData = c_char_p(b'')
    ErrMsg = c_char_p(b'')
    TzLength= c_int()
    nResult = dll.FPIGetFeature(0,TzData,byref(TzLength),ErrMsg)
    #print(len(TzData))
    if nResult==0:
        print("func1:",nResult)
        ImgData=create_string_buffer(31478)
        ImgLength=c_int()
        r=dll.FPIGetImageData(1,ImgData,byref(ImgLength))
        if r==0:
            print("func3:",r)
            #print(repr(ImgData.raw))
            with open(fileName,'wb') as f:
                f.write(ImgData.raw)
        else:
            print("func3:",r)
    else:
        print("func2:",nResult)


getAndSaveImg('10.bmp')
