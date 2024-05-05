# OCR based on CV and a small model

### Setup
-   Create a venv and activate it
    ```
    source venv/bin/activate
    ```

- Install requirements
    ```
    pip install -r requirements.txt
    ```

- Start uvicorn server
    ```
    uvicorn app:app --reload
    ```

- Go to localhost:8000
    ```
    http://localhost:8000/home
    ```
- Put the model h5 files in model folder, obtained from here ->
    [model file](https://drive.google.com/file/d/1pT7yb1RRkgU2jvBPEetv3vNbut8XzJau/view?usp=sharing)
- U can use the application from there

![image](/assignment1/assets/asset1.jpg)
![image2](/assignment1/assets/asset2.jpg)


- Jupyter notebook link which was used to train the model on colab -> [jupyter notebook](https://colab.research.google.com/drive/1v3BUy-0COgRQA8xmVBAVDRw2vNc3zjX1?usp=sharing)