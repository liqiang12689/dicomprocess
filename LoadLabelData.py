#coding:utf-8
#Author：QiangLi
#Date:2019.04.04
'''
input:
    PathDicom:待提取数据的文件路径
output：
    datalist：路径下所有文件的路径列表
    TrainData：路径下所有数据文件
'''

import os
# from LoadFileInformation import *
# from readnii import *
from lib.LoadFileInformation import *
from lib.readnii import *

def max_list(lt):
    # 返回列表中重复最多的一个数据
    temp = 0
    for i in lt:
        if lt.count(i) > temp:
            max_str = i
            temp = lt.count(i)
    return max_str
def LoadSeriesNumber(fileList, path):
    #提取当前患者所有的SeriesNumber
    SeriesNumber = []
    for filename in fileList:
        information = LoadFileInformation(path + '/' + filename)
        # print(information['SeriesInstanceUID'], '%', information['SeriesNumber'])
        SN = information['SeriesNumber']
        SeriesNumber.append(SN)
    return SeriesNumber
def LoadLstFilesDCMSeries(fileList, path, dirName):
    #获取单个患者数据列表
    lstFilesDCMSeries = []
    lstPersionFilesDCMSeries = []
    # SeriesNumber = LoadSeriesNumber(fileList, path)
    for filename in fileList:
        lstFilesDCMSeries = read_img(path + '/' + filename)
    lstPersionFilesDCMSeries.append(lstFilesDCMSeries)
    return lstPersionFilesDCMSeries
def LoadLstPersionFilesDCMSeries(PathDicom, PathsubdirList):
    #获取所有患者数据列表
    lstPersionFilesDCMSeries = []
    for Pathlist in PathsubdirList:
        for dirName, subdirList, fileList in os.walk(PathDicom + '/' + Pathlist):
            lstPersionFilesDCMSeries.append(PathDicom + '/' + Pathlist + '/' + fileList[0])
        # AllPersionFilesDCMSeries.append(lstPersionFilesDCMSeries)
    return lstPersionFilesDCMSeries
def LoadLabelSerices(PathDicom):
    #获取该路径下所有患者列表
    lstPersionFilesDCMSeries = []
    for PathdirName, PathsubdirList, Pathfilename in os.walk(PathDicom):
        if PathsubdirList:
            lstPersionFilesDCMSeries = LoadLstPersionFilesDCMSeries(PathdirName, PathsubdirList)
    return lstPersionFilesDCMSeries

def LoadLabelData(lstPersionFilesDCMSeries):
    #获取所有路径下的数据信息
    LabelData = []
    for pathlist in lstPersionFilesDCMSeries:
        LabelData.append(read_img(pathlist))
    return LabelData
if __name__ == "__main__":
    Labeldatalist = LoadLabelSerices(r'D:\myPython\My2dUnet\unet\data1\train\label')
    Labeldata = LoadLabelData(Labeldatalist)



