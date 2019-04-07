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
from lib.LoadFileInformation import *
from lib.readnii import *
# from LoadFileInformation import *
# from readnii import *

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
    SeriesNumber = LoadSeriesNumber(fileList, path)
    for filename in fileList:
        maxSN = max_list(SeriesNumber)
        information = LoadFileInformation(path + '/' + filename)
        print(path + '/' + filename)
        SN = information['SeriesNumber']
        if SN == maxSN:
            lstFilesDCMSeries.append(os.path.join(dirName, filename))
    lstPersionFilesDCMSeries.append(lstFilesDCMSeries)
    return lstPersionFilesDCMSeries
def LoadLstPersionFilesDCMSeries(PathDicom, PathsubdirList):
    #获取所有患者数据列表
    AllPersionFilesDCMSeries = []
    for Pathlist in PathsubdirList:
        for dirName, subdirList, fileList in os.walk(PathDicom + '/' + Pathlist):
            lstPersionFilesDCMSeries = LoadLstFilesDCMSeries(fileList, PathDicom + '/' + Pathlist, dirName)
        AllPersionFilesDCMSeries.append(lstPersionFilesDCMSeries)
    return AllPersionFilesDCMSeries
def LoadDicomSerices(PathDicom):
    #获取该路径下所有患者列表
    lstPersionFilesDCMSeries = []
    for PathdirName, PathsubdirList, Pathfilename in os.walk(PathDicom):
        if PathsubdirList:
            lstPersionFilesDCMSeries = LoadLstPersionFilesDCMSeries(PathdirName, PathsubdirList)
    return lstPersionFilesDCMSeries

def LoadDicomData(lstPersionFilesDCMSeries):
    #获取所有路径下的数据信息
    TrainData = []
    for pathlist in lstPersionFilesDCMSeries:
        TrainData.append(read_img(pathlist[0]))
    return TrainData
if __name__ == "__main__":
    Traindatalist = LoadDicomSerices(r'D:\myPython\My2dUnet\unet\data1\train\volume')
    TrainData = LoadDicomData(Traindatalist)





