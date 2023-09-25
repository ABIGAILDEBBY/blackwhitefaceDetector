from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_classifier(model, test_dir):
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(86, 86),
        color_mode="rgb",
        shuffle=False,
        class_mode="binary",
        batch_size=32,
        classes=None,
    )

    pred = model.predict(test_generator)
    # Convert probabilities to binary labels (0 or 1)
    pred_labels = (pred > 0.5).astype(int)

    true_labels = test_generator.classes

    accuracy = accuracy_score(true_labels, pred_labels)
    confusion = confusion_matrix(true_labels, pred_labels)
    classification = classification_report(
        true_labels, pred_labels, target_names=["white_faces", "black_faces"]
    )

    return {
        "accuracy": accuracy,
        "confusion_matrix": confusion,
        "classification_report": classification,
    }


# Example usage:
# Load the model and specify the test directory
# model = tf.keras.models.load_model(f"{PATH}/Models/blackVswhite_"+str(epochs)+"_"+str(batch_size)+".h5")
# test_directory = "path/to/test_data_directory"
# results = evaluate_classifier(model, test_directory)
# print("Accuracy:", results['accuracy'])
# print("Confusion Matrix:\n", results['confusion_matrix'])
# print("Classification Report:\n", results['classification_report'])
