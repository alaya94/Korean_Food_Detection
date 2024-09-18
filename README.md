# Korean_Food_Detection
The Script aim to detect korean food and classify it through image .

## Code Structure
- **data/**: This directory contains train data wich is necessary to create the index file.
- **tools/**: This directory contain 3 files:
  - labels.npy: contain the menu classes as labels
  - food_detection_best.pt: Wieghts of Yolov8 mode for food detection
  - Faiss_index.index: File contain storing embeddings of train data as index vectors
 
    ## Running the Application:

1. **Setup Environment**:
   - Create a new virtual Environment by running :
      ```
     python -m venv venv_name 

     ```
   - Connect to the new virtual Environment On Windows:
      ```
     .\venv_name\Scripts\activate

      ```

     On Unix or MacOS:
     ```
     source venv_name/bin/activate      

     ```

   - Ensure that Python and all required dependencies are installed on your Environment. You can typically install dependencies using `pip` by running:

     ```
     pip install -r requirements.txt
     
     ```

2. **Create the index file with Clip and Faiss **:
- You can run the Script by exectue `Create_index.py`:
     ```
     python Create_index.py 

     ```
- You can pass this process if you already has the index file.
  
3. **Test the models **:
    * You can test the model menu classification by running test_classification.py
    * you can test the model menu classification + Yolo detection by running test_detection.py

  

  

  
