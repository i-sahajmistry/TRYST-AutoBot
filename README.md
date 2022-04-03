# Robo-Sapiens- IIT Jammu - AutoBot  (TAB-045)

 - Submission by Team Robo-Sapiens IIT Jammu


## Instructions to run the model.


1 - Clone this repo.
```
git clone https://github.com/i-sahajmistry/Robo-Sapiens-AutoBot.git
```
2 - Create a virtual environment and activate it.
```
python3 -m venv envName
source ./envName/bin/activate
```
3 - Run command to install dependencies.
```
pip install -r requirements.txt
```
4 - Download Model files.
#### Please manually download our Model files of type *.pth from [google drive](https://drive.google.com/drive/folders/11R1dtkgiS13rvqz99jDr5cWBxbvjurrT?usp=sharing) and keep those in root of this repository, where the main.ipynb file is present.
> Since the size of model files are very large (>200MB) it was not possible to push it on github hence we kept those files on google drive.

5 - Move the downloaded files to the folder containing main.ipynb

6 - Run the code

<img src="./images/person.png" height="256">
<img src="./images/animal.png" height="256">
<img src="./images/roadcone.png" height="256">
<img src="./images/zebra.png" height="256">

For the moment, we have only performed the detection of person, animal, roadcone and zebra crossing. We have oly initialized the jetson class instance but not given instructions to run, wait or stop. Detection can be seen in the above mentioned images.
