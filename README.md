# Serving a DL model using Streamlit.

The application uses [Streamlit](https://streamlit.io/) to deploy a computer vision application.

A sample image has been provided for testing, it is available in the root directory as `sample_index_finger.jpg`

To run the application, 
- it is recmmended to create a python virtual environment
```sh
python -m venv env
# This works for windows machine
```
- activate the environment
```sh
env\Scripts\activate
# This works for windows cmd/powershell
source env/Scripts/activate # works with git bash
source env/bin/activate # works with linux/wsl
```
- install app dependencies
```sh
pip install -r requirements.txt

```
- run streamlit
```sh
streamlit run gender_classifier.py
```

The streamlit server will be started and the webpage is opened with a placeholder to upload an image. 
The sample image can be downloaded and uploaded to make predictions with the application.
