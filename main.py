import src.utils as utils
import src.dataset as dataset
import src.model as model
import src.train as train
import src.eval as eval

def main(config):
    """
    
    """
    utils.setup_logging(config["logging"]["file"])
    
    train_loader, test_loader, scaler = dataset.dataset()
    
    lstm_model = model.LSTMModel(
        config["train"]["input_size"], 
        config["train"]["hidden_size"], 
        config["train"]["num_layers"], 
        config["train"]["output_size"])
    
    train.train_lstm_model(lstm_model, train_loader, config)

    predictions, actuals = eval.evaluate(lstm_model, test_loader, config)

    eval.plot_predictions(dataset.data, predictions, actuals, scaler)

if __name__ == "__main__":
    config_path = "./configs/config.yaml"
    config = utils.load_config(config_path)
    main(config)

