import re
import os
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


# Authentication + token creation
def create_drive_manager(cmd_auth):
    gAuth = GoogleAuth()

    if cmd_auth:
        gAuth.CommandLineAuth()
    else:
        gAuth.LocalWebserverAuth()

    gAuth.Authorize()
    print("authorized access to google drive API!")

    drive: GoogleDrive = GoogleDrive(gAuth)
    return drive

    
def extract_files_id(drive, link):
    try:
        fileID = re.search(r"(?<=/d/|id=|rs/).+?(?=/|$)", link)[0]  # extract the fileID
        return fileID
    except Exception as error:
        print("error : " + str(error))
        print("Link is probably invalid")
        print(link)


def pydrive_download(drive, link, save_path):
    id = extract_files_id(drive, link)
    file_dir = os.path.dirname(save_path)
    if file_dir:
        os.makedirs(file_dir, exist_ok=True)

    pydrive_file = drive.CreateFile({'id': id})
    pydrive_file.GetContentFile(save_path)
