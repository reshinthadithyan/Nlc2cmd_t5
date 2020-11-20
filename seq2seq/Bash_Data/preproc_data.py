import json
from tqdm import tqdm
import random

def Convert_Data(Path):
    """Data to T5 Training"""
    Json = read_json(Path)
    Dataset = []
    for i in tqdm(range(1,len(Json)+1)):
            data = Json[str(i)]
            NL = data["invocation"]
            Cmd =data["cmd"]
            Data = NL + "#@#" + Cmd 
            Dataset.append(Data)  
    return Dataset

def Split_Data(Dataset):
    random.shuffle(Dataset)
    Length = len(Dataset)
    Trainset = Dataset[:int(Length*0.9)]
    Valset  = Dataset[int(Length*0.9):int(Length*0.95)]
    Testset = Dataset[int(Length*0.95):int(Length)]
    return Trainset,Valset,Testset

def Write_Data(Dataset,File):
    Src_File = File + ".source"
    Tar_File = File + ".target"
    Src_List = []
    Tar_List = []
    for i in Dataset:
        s,t = i.split("#@#")
        Src_List.append(s)
        Tar_List.append(t)
    with open("G:\Work Related\Transformers\examples\seq2seq\Bash_Data\T5_Data/"+Src_File, "w",encoding="utf-8") as outfile:
        outfile.write("\n".join(Src_List))
    with open("G:\Work Related\Transformers\examples\seq2seq\Bash_Data\T5_Data/"+Tar_File, "w",encoding="utf-8") as outfile:
        outfile.write("\n".join(Tar_List))
    return "Done"

def read_json(jsonpath):
    with open(jsonpath) as json_file:
        data = json.load(json_file)
    return data

if __name__ == "__main__":
    BashData = Convert_Data(r"G:\Work Related\Transformers\examples\seq2seq\Bash_Data\nl2bash-data.json")
    Trainset,Valset,Testset = Split_Data(BashData)
    Write_Data(Trainset,"train")
    Write_Data(Valset,"val")
    Write_Data(Testset,"test")