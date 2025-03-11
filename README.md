# :fire: Diddy Learning Project :fire:

## :zap: How To Install this project and Train by yourself :zap:

> * :milky_way: [Train_and_use_app](#install-python)
> * :crystal_ball: [Use_app_only](#use-model-on-localhost)

### Install Python
> *[Teleport](#fire-diddy-learning-project-fire)<br>
    # We will install python 3.10 by anaconda software. You can download form link: [Dowload](https://www.anaconda.com/download)

### Create Environment
> *[Teleport](#fire-diddy-learning-project-fire)
```
    > conda create -y --name dl_env python=3.10
    > conda activate dl_env
    > conda install -y -c conda-forge cudatoolkit=11.8 cudnn=8.9.7 cudatoolkit-dev

    # Tensorflow - Anything above 2.10 is not supported on the GPU on Windows Native
    > python -m pip install "tensorflow<2.11"  "tensorflow-text<2.11"  "numpy<2.0" 

    # Verify Tensorflow GPU installation
    > python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

    # PyTorch
    > pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118

    # Verify PyTorch GPU installation
    > python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"

    > pip install Pillow scikit-learn numpy pandas matplotlib pandas seaborn missingno nltk graphviz pydot imutils tensorflow-datasets datasets transformers "notebook==6.5.4"

    > pip install --upgrade tensorflow-datasets plotly ipykernel ipyflow jupyter_contrib_nbextensions

    > ipython kernel install --user --name=dl_env

    > python -m ipykernel install --user --name dl_env --display-name dl_env

    # สำหรับระบบ Linux หลังจากที่ activate dl_env แล้ว ควรรันคำสั่ง
    ~$ export PATH=$CONDA_PREFIX/bin:$PATH
    ~$ export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

    # รัน jupyter notebook ดัวยคำสั่ง
    > jupyter notebook หรือ python -m notebook
```

### Choose your environment that you created and install these libraries
> *[Teleport](#fire-diddy-learning-project-fire)
```
    # Open your anaconda prompt select your environment by type conda activate dl_env. If you want to check list of your environment you can type conda env list.
    > pip install ultralytics opencv-python numpy matplotlib

    Congratulation 🎉🎉🎉 you have setting succesfully.
```

### ZIP
> *[Teleport](#fire-diddy-learning-project-fire)<br>
- Extract your zip to location that you want to install
![This is picture.](/Image_step_project/afterExtractZip.png "Go Go GOD!!!")
- [optional]You can see all file in Visual Studio Code to.
![This is picture.](/Image_step_project/inVisualStudioCode.png "Hi VS-Code")

### Training Time!!!
> *[Teleport](#fire-diddy-learning-project-fire)
- Zero, press ctrl+shift+p click select interpreter and choose the name of environment that you crate in [Create_Environment](#create-environment)
- First, observation content path in data.yaml.
![This is picture.](/Image_step_project/datadotyam.png "yummy")
- Second, setting your path in train, val and test.
![This is picture.](/Image_step_project/changePath.png "change it")
- Third, setting your path in model.train().
![This is picture.](/Image_step_project/modelTraining.png "change it")
    - Third_one: Change __data__ parameter form '../dataset/data.yaml' to your data.yaml file location.
    - Third_two: Change your __project__ parameter from 'D:\\AB-BiMaGOoOD\\DiddyLearning\\model\\runs\\detect' to the location that you want to save the result of training model. In addition, you can change __name__ parameter too because the result of training model will save at your desire location with name that you set.
- Forth, setting your path in model.eval().
![This is picture.](/Image_step_project/modelEval.png "change it")
    - Forth_one: Change path file in YOLO(path/of/your/best.pt)<br>
    ** best.pt will be in every train file, if you training model n times, train file will increase the number too except we set 'exist_ok = true'.**
    - Forth_two: Change path file in model.val(imgsz=640, project='path\\to\\your\\dest\\location',conf=0.313)<br>
    ** project is mean we want to save the result file in location what we want.**<br>
    ** you can reference this value from 'model/runs/detect/train2' and look in F1_curve.png and you will see all classes at {floating_number:.3f}
- Fifth, setting your path in model.prediction().
![This is picture.](/Image_step_project/predict.png "change it")
    - Fifth_one: Change _predictImagexx_ from the location that you want to pull image for prediction.
    - Fifth_two: You can change __train2__ in path to another train YOLO("../model/runs/detect/train2/weights/best.pt").<br>
    ** train2 is mean the model has been run for 2 rounds.**
    - Fifth_three: model([predict_image_01, predict_image_02, predict_image_03, predict_image_04], conf= 0.342)<br>
    ** _conf_ stands for confidence value and you can reference this value from 'model/runs/detect/val/val2' and look in F1_curve.png and you will see all classes at {floating_number:.3f}<br>
    ** The conf value is a value that tells us that when the conf value is at this point, it will be the point that makes the F1 - score of all classes the highest.So, conf value is like a threshold.
    ** val2 is like a train2 is mean the model has been run model.val for 2 rounds.**
- sixth, setting your path in model.prediction more.
![This is picture.](/Image_step_project/lastPredict.png "change it")
    - Sixth_one: set your images test path.

### Congrattt 🎉🎉🎉 now you can run Import Library, Model Training, Evaluate Model and Predicted by Model. I don't recommand you to run Data and Pre-Processing because it's not have data to change anymore or original dataset.

### Use model on localhost
> *[Teleport](#fire-diddy-learning-project-fire)
- first, you should go to this [Link](https://github.com/potaege/web-detected-Parasitic-Egg.git)
- second, use git clone "https://github.com/potaege/web-detected-Parasitic-Egg.git" in git bash or command prompt
- third, use dl_env or use environment that you create in [Create_Environment](#create-environment) to install these libraries.
- Forth, install python and then when python installed. Use pip install Flask ultralytics opencv-python when you successfully installing complete.
- Fifth, open the file in VS-Code and press ctrl+shift+p click select interpreter and choose the name of environment that you crate in [Create_Environment](#create-environment)
- Sixth, click terminal and choose new terminal and then type python app.py to run our project application.
- [Learn more](https://github.com/potaege/web-detected-Parasitic-Egg.git)

### HandBook about YOLO - YOU ONLY LOOK ONCE
> *[Teleport](#fire-diddy-learning-project-fire)
- [Preprocessing Annotated Data](https://docs.ultralytics.com/guides/preprocessing_annotated_data/#resizing-images)
- [Model Modes](https://docs.ultralytics.com/modes/train/#apple-silicon-mps-training)

### Thank you for reading. I hope you can do it from our guide. If our guide misleads you, is not easy to understand, or contains incorrect information, we apologize here for sincerely. Thank you 🙏🙏🙏
