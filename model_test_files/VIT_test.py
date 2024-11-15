
from transformers import AutoModelForImageClassification, pipeline
def print_hi(input):
    #input文件名字
    # Use a breakpoint in the code line below to debug your script.
    checkpoint = "../Models/train_with_3666_vit/checkpoint-171"


    model = AutoModelForImageClassification.from_pretrained(
        checkpoint
    )
    classifier_2 = pipeline(
        task="image-classification", model=model, image_processor="../Models/train_with_3666_vit/checkpoint-171/preprocessor_config.json"
    ) # Press Ctrl+F8 to toggle the breakpoint.

    ##结果
    print(classifier_2(input))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    img = '../test_images/11.jpg'
    print_hi(img)

