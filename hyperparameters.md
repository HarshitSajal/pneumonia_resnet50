# Hyperparameter Choices

Learning Rate: `1e-4`  
Chosen as a moderate starting point for fine-tuning. Lower values help avoid catastrophic forgetting of pretrained features.

Batch Size: `32`  
Balanced choice for training speed and generalization.

Epochs: `50`  
Chosen to give the model ample time to converge, with early stopping used to prevent overfitting.

Early Stopping Patience: `2`  
Stops training when validation accuracy doesn't improve, which helps mitigate overfitting.

Optimizer: `Adam`  
Chosen for its adaptive learning rate and effectiveness in deep learning fine-tuning tasks.

Loss Function: `CrossEntropyLoss`  
Standard for multi-class classification.
