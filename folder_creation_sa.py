import os
import sys
import subprocess
import shutil
import sys
from distutils.dir_util import copy_tree
import mysql.connector
import logging
import schedule
import time
import googlemaps
logging.basicConfig(filename='FolderCreationLog.log',
                    level=logging.DEBUG, format='%(asctime)s %(message)s')


# when executing script, 4 variables need to be passed to it. These are estimate_id, job_type & project_type and address in that order.
#  eg, ./script.py 0000001 Variation BASIX

#####################################################
# Create new folder in /Volumes/PCIe/Dropbox/CE Technical Team/.
# Subfolder based on a combination of job_type. Move items in particular estimate folder to
###
### production ###
###                             ###
#####################################################

#############
# 09/10/2021
# find how to create VIP clients folder directly 
# logic - when VIP name found from database(subject)
# categorise folders in VIP name not in project name
#############


def job():
    print("Running automation")
    schedule.every(300).seconds.do(main)

    while 1:
        schedule.run_pending()
        time.sleep(1)


def move_estimate_folder(project_type, estimate_id, estimate_folder_path, production_location, address):

    commercial_template = "/Users/admin/Desktop/TEST/Volumes/PCIe/CE General/1 QA Documents/Production/01 Job Template Folders/Commercial/* QA Section J Commercial - Job Template Folder"
    residential_template_dts = "/Users/admin/Desktop/TEST/Volumes/PCIe/CE General/1 QA Documents/Production/01 Job Template Folders/Residential/*QA DTS Residential"
    residential_template = "/Users/admin/Desktop/TEST/Volumes/PCIe/CE General/1 QA Documents/Production/01 Job Template Folders/Residential/*QA Residential - Job Template Folder"
    # empty template, need to fill up
    residential_template_mixed_boarding_house = "/Users/admin/Desktop/TEST/Volumes/PCIe/CE General/1 QA Documents/Production/01 Job Template Folders/Residential/*QA Mixed Use/Boarding House"
    # 19.07.2021 SUNG - new edited template for WSUD
    residential_template_WSUD = "/Users/admin/Desktop/TEST/Volumes/PCIe/CE General/1 QA Documents/Production/01 Job Template Folders/Residential/* QA WSUD - Job Template Folder"
    source = estimate_folder_path

    residential_r2 = ["NatHERS + BASIX", "BASIX and/or Section J", "BASIX", "NatHERS", "BESS", "BASIX DIY", "6 Star Energy Efficiency Report", "NatHERS, BASIX, & Section J",
                      "ESD", "Energy Efficiency Report", "SDA", "NatHERS + BASIX + Section J", "Energy Efficiency Report + Section J", "VURB", "Section J + BASIX",
                      "WSUD"]
    commercial_c7 = ["NABERS", "NABERfS : Energy Rating",
                     "NABERS : Preliminary Strategy Report"]

    if project_type in residential_r2:
        print("Checkpoint 1")
        project_folder = production_location + \
            "/ *Residential/*NatHERS jobs/" + address + " " + estimate_id
        print("Checkpoint 2")
        os.mkdir(project_folder)
        # 24.08.21 SUNG - check FileExistsError if Y delete existing folder with same name
        print("Checkpoint 3")
        copy_tree(source, project_folder)
        print("Checkpoint 4")
        if project_type == "6 Star Energy Efficiency Report":
            copy_tree(residential_template_dts, project_folder)
        elif project_type == "Section J + BASIX":
            # no template exist for residential_template_mixed_boarding_house, replace to residential_template for temporarily
            # original code - copy_tree(residential_template_mixed_boarding_house, project_folder)
            copy_tree(residential_template, project_folder)
        # 19.07.2021 SUNG - new edited line for WSUD
        elif project_type == "WSUD":
            copy_tree(residential_template_WSUD, project_folder)
        else:
            copy_tree(residential_template, project_folder)
        # copy project source files to Residential/NatHERS jobs(R2)
        print("Checkpoint 5")
        folder_logic(project_folder, estimate_id)
        return project_folder

    elif project_type == "DTS Energy Efficiency Report":
        print("Checkpoint 1")
        project_folder = production_location + \
            "/ *Residential/*DTS Residential Jobs/" + address + " " + estimate_id
        print("Checkpoint 2")
        os.mkdir(project_folder)
        print("Checkpoint 3")
        copy_tree(source, project_folder)
        print("Checkpoint 4")
        copy_tree(residential_template_dts, project_folder)
        print("Checkpoint 5")
        folder_logic(project_folder, estimate_id)
        return project_folder
        # move it to Residential/DTS Energy Efficiency Report(R1)

    elif "Green Star" in project_type:
        project_folder = production_location + \
            "/ *Commercial/-Greenstar/" + address + " " + estimate_id
        os.mkdir(project_folder)
        copy_tree(source, project_folder)
        return project_folder
        # move it to Commercial/Greenstar(C3)

    elif project_type == "Section J":
        project_folder = production_location + \
            "/ *Commercial/-DTS Commercial Jobs/" + address + " " + estimate_id
        os.mkdir(project_folder)
        copy_tree(source, project_folder)
        copy_tree(commercial_template, project_folder)
        #print("checkpoint 9")
        folder_logic(project_folder, estimate_id)
        return project_folder
        # move it to Commercial/DTS Commercial Jobs(C2)

    elif project_type == "JV3":
        project_folder = production_location + \
            "/ *Commercial/-JV3 Jobs/" + address + " " + estimate_id
        os.mkdir(project_folder)
        copy_tree(source, project_folder)
        copy_tree(commercial_template, project_folder)
        folder_logic(project_folder, estimate_id)
        return project_folder
        # move it to Commercial/JV3 Jobs(C4)

    elif project_type == "WELL":
        project_folder = production_location + \
            "/ *Commercial/-Well/" + address + " " + estimate_id
        os.mkdir(project_folder)
        copy_tree(source, project_folder)
        return project_folder
        # move it to Commercial/Well(C8)

    elif project_type == "BEEC":
        project_folder = production_location + \
            "/ *Commercial/-BEEC/" + address + " " + estimate_id
        os.mkdir(project_folder)
        copy_tree(source, project_folder)
        folder_logic(project_folder, estimate_id)
        return project_folder
        #copy_tree(residential_template, project_folder)
        # move it to Commercial/BEEC(C9)

    elif project_type == "CFD Modelling":
        project_folder = production_location + \
            "/ *Commercial/-CFD Modelling/" + address + " " + estimate_id
        os.mkdir(project_folder)
        copy_tree(source, project_folder)
        return project_folder
        # move it to Commercial/CFD Modelling(C1)

    elif "Nabers" in project_type:
        project_folder = production_location + \
            "/ *Commercial/-Nabers/" + address + " " + estimate_id
        os.mkdir(project_folder)
        copy_tree(source, project_folder)
        return project_folder
        # move it to Commercial/Nabers(C7)

    elif project_type == "Thermal Comfort":
        project_folder = production_location + \
            "/ *Commercial/-Thermal comfort/" + address + " " + estimate_id
        os.mkdir(project_folder)
        copy_tree(source, project_folder)
        return project_folder
        # move it to Commercial/Thermal comfort(C5)

    elif project_type == "Life Cycle Analysis":
        project_folder = production_location + \
            "/ *Commercial/-LifeCycle/" + address + " " + estimate_id
        os.mkdir(project_folder)
        copy_tree(source, project_folder)
        return project_folder
        # move it to Commercial/-LifeCycle(C10)

    elif project_type == "DTS":
        project_folder = production_location + \
            "/ *Commercial/-DTS Commercial Jobs/" + address + " " + estimate_id
        os.mkdir(project_folder)
        copy_tree(source, project_folder)
        folder_logic(project_folder, estimate_id)
        return project_folder
        # move it to Commercial/DTS Commercial Jobs(C2)

    else:
        logging.info("Job type mismatch. Not in db. estimate# " + estimate_id)


def find_estimate_folder(estimate_id, estimate_location):
    location = estimate_location
    for root, subdirs, files in os.walk(location):
        for d in subdirs:
            if estimate_id in d:
                logging.info("Found estimate location: " + d)
                return os.path.join(root, d)


def folder_logic(project_folder, estimate_id):
    #print("checkpoint 10")
    for root, subdirs, files in os.walk(project_folder):
        for f in files:
            if "Fee_Proposal" in f:
                print("foundestimate_pdf")
                src = os.path.join(root, f)
                dest = os.path.join(root, "XXXXXX 0. Fee Proposal")
                shutil.move(src, dest)

            elif ".DS_Store" not in f:
                src = os.path.join(root, f)
                dest = os.path.join(root, "XXXXXX 1. Plans from Client")
                shutil.move(src, dest)

        for d in subdirs:
            if estimate_id in d:
                src = os.path.join(root, d)
                dest = os.path.join(root, "XXXXXX 1. Plans from Client/" + d)
                shutil.move(src, dest)
                # return os.path.join(root, d)

        for d in subdirs:
            if "XXXXXX" in d:

                d_path = os.path.join(root, d)

                for root1, subdirs1, files1 in os.walk(d_path):

                    for fd in files1:
                        if "XXXXX" in fd:
                            src = os.path.join(root1, fd)
                            dest = os.path.join(
                                root1, fd.replace("XXXXXX", estimate_id))
                            shutil.move(src, dest)

                    for sd in subdirs1:
                        if "XXXXX" in sd:
                            src = os.path.join(root1, sd)
                            dest = os.path.join(
                                root1, sd.replace("XXXXXX", estimate_id))
                            shutil.move(src, dest)

                src = os.path.join(root, d)
                dest = os.path.join(root, d.replace("XXXXXX", estimate_id))
                shutil.move(src, dest)


def fetch_estimates():
    # print("checkpoint1")
    mydb = mysql.connector.connect(
        host="115.70.228.70",
        user="dronegp_deal",
        passwd="4ZyhNrR5lEhJf4Fe",
        database="dronegp_deals"
    )

    mycursor = mydb.cursor()

    mycursor.execute(
        "SELECT id, subject, estimate_number from harvest_estimates WHERE deal_id > 0 AND accepted = 1 AND ptbp_error IS NULL AND production_folder = 0 order by id desc")

    myresult = mycursor.fetchall()

    if myresult:
        for x in myresult:
            print(x[2])
            print(x[1].split(" - "))
            # print("checkpoint2.")
    else:
        print("None Found")
        logging.info("No estimates to process")

    # print("checkpoint2")
    return myresult


def updated_db(production_folder_status, estimate_id):

    mydb = mysql.connector.connect(
        host="115.70.228.70",
        user="dronegp_deal",
        passwd="4ZyhNrR5lEhJf4Fe",
        database="dronegp_deals"
    )

    mycursor = mydb.cursor()

    sql = "UPDATE `harvest_estimates` SET `production_folder` = %s WHERE estimate_number = %s"
    val = (production_folder_status, str(estimate_id))

    mycursor.execute(sql, val)
    mydb.commit()
    result = mycursor.rowcount, "record(s) affeceted"
    logging.info(result)


def main():

    # prod location =  /Volumes/PCIe/Dropbox/CE Technical Team/ * Residential or *Commercial
    # estimate_location = "/Volumes/PCIe/CE General/2 Sales & Marketing/1 Quoting/**Estimates"
    # production_location = "/Volumes/PCIe/Dropbox/CE Technical Team/Automation_Testing_New/CE Technical Team"
    # template_location = "/Users/admin/Desktop/TEST/Volumes/PCIe/CE General/1 QA Documents/Production/01 Job Template Folders/"

    estimate_location = "/Volumes/CE General/2 Sales & Marketing/1 Quoting/**Estimates"
    
    # previous path
    # production_location = "/Volumes/CE Technical Team/Automation_Testing_New/CE Technical Team"
    # 26.08.21 testing local repository and worked fine
    # production_location = "/Users/admin/Desktop/Automation_Testing_New/CE Technical Team"
    # Try above when below directory gives error
    
    production_location = "/Volumes/CE Technical Team/Automation"
    template_location = "/Users/admin/Desktop/TEST/Volumes/PCIe/CE General/1 QA Documents/Production/01 Job Template Folders/"

    print("Fetching estimates from db")
    logging.info("Fetching estimates from db")
    fetched_estimates = fetch_estimates()

    # print("checkpoint3")
    for es in fetched_estimates:
        try:
            # print("checkpoint4")
            split = es[1].split(" - ")
            # print("checkpoint5")
            estimate_id = es[2]
            # print("checkpoint6")
            job_type = split[1]
            # print("checkpoint7")
            address = split[2]
            # print("checkpoint8")

        except:
            print("Subject formatting error/No estimates in need of processing ")
            logging.info(
                "Subject formatting error/No estimates in need of processing ")
            production_folder_status = 10
            updated_db(production_folder_status, estimate_id)

        try:
            print("trying to find estimate#" + estimate_id)
            logging.info("trying to find estimate#" + estimate_id)
            estimate_folder_path = find_estimate_folder(
                estimate_id, estimate_location)
            if estimate_folder_path:
                print("Creating prod folder for estimate# " + estimate_id)
                print("Finding estimate foldder for estimate# " + estimate_id)
                print(
                    "Creating production files, copying template for estimate# " + estimate_id)
                logging.info(
                    "Creating prod folder for estimate# " + estimate_id)
                logging.info(
                    "Finding estimate foldder for estimate# " + estimate_id)
                logging.info(
                    "Creating production files, copying template for estimate# " + estimate_id)

                project_folder = move_estimate_folder(
                    job_type, estimate_id, estimate_folder_path, production_location, address)
                print("Running sa with project path: " + project_folder)
                logging.info("Running sa with project path: " + project_folder)
                run_sa = 'python3 site_analysis.py "' + project_folder + '" "' + address + '"'
                sa_status = subprocess.run(run_sa, shell=True)

                production_folder_status = 1
                print("Production folder created for estimate# " + estimate_id)
                logging.info(
                    "Production folder created for estimate# " + estimate_id)

                updated_db(production_folder_status, estimate_id)
                print("Updated db prod folder status for estimate# " + estimate_id)
                print("Created successfully")
                print("#########################")
                logging.info(
                    "Updated db prod folder status for estimate# " + estimate_id)
                logging.info("Created successfully")
                logging.info("#########################")

            else:
                print("estimate not found")
                logging.info("estimate not found")
                production_folder_status = 11
                updated_db(production_folder_status, estimate_id)

        except Exception as ex:
            #when address can't be found from google maps to genereate site analysis
            production_folder_status = 9
            updated_db(production_folder_status, estimate_id)
            print("Exception: " + ex)
            print(
                "Possible duplicate, production folder not created for estimate# " + estimate_id)
            logging.info(
                "Possible duplicate, production folder not created for estimate# " + estimate_id)
            print(ex)

    logging.info("End Run")
    logging.info("#########################")
    print("End Run")
    print("#########################")


if __name__ == '__main__':
    job()
