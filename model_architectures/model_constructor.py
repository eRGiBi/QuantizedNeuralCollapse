import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as torchvision_models
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

from model_architectures.convnext_nano import ConvNeXtNano
from model_architectures.simple_cnn import SimpleCNN


class ModelWrapper(nn.Module):
    """Make standard torchvision models compatible with the NC analyzer.
    """
    def __init__(self, base_model, num_classes, feature_dim=512):
        super().__init__()
        # Get all layers except the original fully connected one
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        
        # The new penultimate layer
        in_features = base_model.fc.in_features
        self.penultimate = nn.Linear(in_features, feature_dim)
        
        # The final classifier
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.penultimate(x))
        x = self.classifier(x)
        return x

        
class ModelConstructor:
    """A class to construct models based on specified architecture names."""
    
    @staticmethod
    def get_model(config):
        """
        Selects, constructs, and returns the specified model.
        
        Returns:
            torch.nn.Module: The constructed model.
        """
        if config["task"] == "cv":
            input_channels = 1 if config["dataset"].upper() == 'MNIST' else 3

            match config["model"].lower():
                
                case 'simple_cnn':

                    return SimpleCNN(
                        num_classes=config["num_classes"],
                        #  input_channels=input_channels
                        ).to(config["device"]
                    )

                case 'mobilenet':
                    model = torchvision_models.mobilenet_v3_large(
                        weights=None if not config["pretrained"] else
                            torchvision_models.MobileNet_V3_Large_Weights.IMAGENET1K_V2,
                        progress=True,
                        kwargs = {
                            "stochastic_depth_prob": 0.0
                        }
                    )
                    in_features = model.classifier[-1].in_features
                    model.classifier[-1] = nn.Linear(in_features, config["num_classes"])

                    return model

                case 'convnextsmall':
                    model = torchvision_models.convnext_small(
                        weights=None if not config["pretrained"] else
                        torchvision_models.ConvNeXt_Small_Weights.IMAGENET1K_V1,
                        progress=True,
                        kwargs = {
                            "stochastic_depth_prob": 0.0
                        }
                    )

                    in_features = model.classifier[-1].in_features
                    model.classifier[-1] = nn.Linear(in_features, config["num_classes"])

                    return model

                case 'convnexttiny':
                    model = torchvision_models.convnext_tiny(
                        weights=None if not config["pretrained"] else
                            torchvision_models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
                        progress=True
                    )

                    in_features = model.classifier[-1].in_features
                    model.classifier[-1] = nn.Linear(in_features, config["num_classes"])

                    return model

                case 'convnextnano':
                    return ConvNeXtNano(
                        num_classes=config["num_classes"],
                        #  input_channels=input_channels
                    ).to(config["device"]
                 )
    
                case 'resnet18':
                    
                    print("Using torchvision ResNet-18 model.")
                    base_model = torchvision_models.resnet18(weights=None) # weights=None for training from scratch

                    # Adaption if necessary
                    if input_channels == 1:
                        base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                        
                    return ModelWrapper(base_model, num_classes=config["num_classes"])
            
        elif config["task"] == "nlp":
            
            match config["model"].lower():
                
                case "gpt2":
                
                    # Load model and tokenizer
                    print("Loading GPT2 model...")
                    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                    tokenizer.pad_token = tokenizer.eos_token

                    model = GPT2LMHeadModel.from_pretrained("gpt2")
                    model.to(config["device"])
                    
                    return model, tokenizer

        else:
            raise ValueError(f"Model '{config['model']}' not supported.")
