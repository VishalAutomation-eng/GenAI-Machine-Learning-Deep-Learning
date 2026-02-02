from data.dataset import load_flowers102
from models.resnet50_model import build_resnet50
from models.vgg16_model import build_vgg16
from models.mobilenetv2_model import build_mobilenetv2
from training.train import train_model
from evaluation.evaluate import evaluate_model
from visualization.plot_history import plot_history

def main():
    train_ds, val_ds, test_ds, info = load_flowers102()
    print(info)

    models = {
        "resnet50": build_resnet50(),
        "vgg16": build_vgg16(),
        "mobilenetv2": build_mobilenetv2(),
    }

    for name, model in models.items():
        print(f"\nTraining {name.upper()}...")
        history = train_model(model, train_ds, val_ds, name)
        plot_history(history, name)
        evaluate_model(model, test_ds, name)
        model.save(f"{name}_flowers102.h5")

if __name__ == "__main__":
    main()
