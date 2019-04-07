import pydicom
import os
from getInput import LoadFileInformation
from readnii import read_img

PathDicom = "./dicom"
lstFilesDCMSeries2 = []  # create an empty list
lstFilesDCMSeries1 = []
for dirName, subdirList, fileList in os.walk(PathDicom):
    information = LoadFileInformation.LoadFileInformation(PathDicom + '/' + fileList[0])
    SeriesNumber1 = information['SeriesNumber']
    for filename in fileList:
        information = LoadFileInformation.LoadFileInformation(PathDicom + '/' + filename)
        print(information['SeriesInstanceUID'],'%',information['SeriesNumber'])
        SeriesNumber2 = information['SeriesNumber']
        if "DCM" in information['CodingSchemeDesignator']:
            if SeriesNumber2 == SeriesNumber1:
                lstFilesDCMSeries1.append(os.path.join(dirName, filename))
            else:
                lstFilesDCMSeries2.append(os.path.join(dirName, filename))

# Get ref file
lstFilesDCM = []
if len(lstFilesDCMSeries2) > len(lstFilesDCMSeries1):
    RefDs = pydicom.read_file(lstFilesDCMSeries2[0])
    lstFilesDCM = lstFilesDCMSeries2
else:
    RefDs = pydicom.read_file(lstFilesDCMSeries1[0])
    lstFilesDCM = lstFilesDCMSeries1
location = []
for i in range(len(lstFilesDCM)):
    data = read_img(lstFilesDCM[i])
    location.append(data)
