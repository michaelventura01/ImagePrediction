from imageai.Prediction.Custom import CustomImagePrediction, ModelTraining
def ImageML():
    prediction = CustomImagePrediction()
    prediction.setModelTypeAsResNet()
    prediction.setModelPath("idenprof/models/idenprof_061-0.7933.h5")
    prediction.setJsonPath("idenprof/json/model_class.json")
    prediction.loadModel(num_objects=10)
    predictions, probabilities = prediction.predictImage("image.jpg", result_count=3)
    for eachPrediction, eachProbability in zip(predictions, probabilities):
        print(eachPrediction, " : ", eachProbability)

def ImageREcognition():
    model_trainer = ModelTraining()
    model_trainer.setModelTypeAsResNet()
    model_trainer.setDataDirectory("idenprof")
    model_trainer.trainModel(num_objects=10, num_experiments=100, enhance_data=True, batch_size=32,
                             show_network_summary=True)

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ImageML()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
