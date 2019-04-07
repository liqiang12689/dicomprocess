import pydicom

def LoadFileInformation(filename):
    information = {}
    # print(filename)
    ds = pydicom.read_file(filename)
    information['w'] = ds.Rows
    information['h'] = ds.Columns
    information['type'] = ds.Modality
    information['PatientBirthDate'] = ds.PatientBirthDate
    information['PatientSex'] = ds.PatientSex
    information['SeriesNumber'] = ds.SeriesNumber
    information['SeriesInstanceUID'] = ds.SeriesInstanceUID
    information['SilceLcation'] = ds.SliceLocation
    information['StudyID'] = ds.StudyID
    information['StudyDate'] = ds.StudyDate
    information['StudyTime'] = ds.StudyTime
    information['SOPInstanceUID '] = ds.SOPInstanceUID
    information['Manufacturer'] = ds.Manufacturer
    information['CodingSchemeDesignator'] = ds.DerivationCodeSequence[0].CodingSchemeDesignator

    # print (dir(ds))
    # print (type(information))
    return information

