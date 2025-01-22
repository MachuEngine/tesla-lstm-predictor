import src.utils as utils
import src.dataset as dataset
import src.model as model
import src.train as train
import src.eval as eval
import src.visualization as visualization

def main(config):
    """
    
    """
    utils.setup_logging(config["logging"]["file"])
    
    train_loader, test_loader, close_scaler, volume_scaler, data = dataset.dataset()
    
    lstm_model = model.LSTMModel(
        config["train"]["input_size"], 
        config["train"]["hidden_size"], 
        config["train"]["num_layers"], 
        config["train"]["output_size"])
    
    train.train_lstm_model(lstm_model, train_loader, config)

    predictions, actuals = eval.evaluate(lstm_model, test_loader, config)

    visualization.plot_predictions(predictions, actuals, close_scaler, volume_scaler, data)

if __name__ == "__main__":
    config_path = "./configs/config.yaml"
    config = utils.load_config(config_path)
    main(config)

