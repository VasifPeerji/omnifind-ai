from sentence_transformers import SentenceTransformer, InputExample, losses, SentenceTransformerTrainer
from torch.utils.data import DataLoader


model = SentenceTransformer("intfloat/e5-base-v2")

train_examples = [
    # Positive pairs (label=1.0)
    InputExample(texts=["red party dress women", "women's evening party dress red"], label=1.0),
    InputExample(texts=["running shoes", "men's sports running sneakers"], label=1.0),
    InputExample(texts=["leather wallet", "brown genuine leather men's wallet"], label=1.0),

    # Negative pairs (label=0.0, unrelated)
    InputExample(texts=["red dress", "red sports tshirt for women"], label=0.0),
    InputExample(texts=["running shoes", "office formal shoes"], label=0.0),
]


train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=2)

train_loss = losses.CosineSimilarityLoss(model)

trainer = SentenceTransformerTrainer(
    model=model,
    train_dataset=train_examples,
    loss=train_loss,
    train_dataloader=train_dataloader,
    epochs=1,
    output_dir="fine_tuned_model"
)

trainer.train()

model.save("fine_tuned_model")
