from transformers import VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, ViTImageProcessor, AutoTokenizer
from datasets import Dataset
from PIL import Image

import torch
import pandas as pd
import os

# Bagian preprocessing dataset Flickr8k nya (Berhasil)

caption_file = r"C:\Users\FELIX\Downloads\archive\captions.txt" # Bisa diganti sama path dari dataset Flickr8k masing-masing
image_folder = r"C:\Users\FELIX\Downloads\archive\Images" # Bisa diganti sama path dari dataset Flickr8k masing-masing

processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

df = pd.read_csv(caption_file, sep=",", encoding="utf-8", on_bad_lines="skip")
df.columns = df.columns.str.strip()

df_grouped = df.groupby('image')['caption'].apply(list).reset_index()

def preprocess_data(example):
    if isinstance(example["caption"], list):
        captions = " ".join([str(c) if c is not None else "" for c in example["caption"]])
    else:
        captions = str(example["caption"]) if example["caption"] is not None else ""

    image_path = os.path.join(image_folder, example["image"])
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.squeeze(0)

    tokenized_caption = tokenizer(
        captions,
        padding="max_length",
        truncation=True,
        max_length=32,
        return_tensors="pt"
    )

    return {
        "pixel_values": pixel_values,
        "input_ids": tokenized_caption["input_ids"].squeeze(0),
        "attention_mask": tokenized_caption["attention_mask"].squeeze(0),
    }

dataset = Dataset.from_pandas(df_grouped)
dataset = dataset.map(preprocess_data, remove_columns=["image", "caption"])

dataset = dataset.train_test_split(test_size=0.1)

# Bagian training pakai Flickr8k untuk ningkatin akurasi (fine-tune) model ViT-GPT-2 nya
''' Catatan : Masih gagal + sering bikin ngelag + waktu udah mepet,
    jadi kita pakai model ViT-GPT-2 yang udah
    di-fine-tune pakai Flickr8k sama orang lain di website HuggingFace
    untuk hasilkan deskripsi gambar di aplikasi nya '''

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class CustomDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features):
        pixel_values = torch.stack([f["pixel_values"] for f in features])
        input_ids = torch.stack([f["input_ids"] for f in features])
        attention_mask = torch.stack([f["attention_mask"] for f in features])

        text_batch = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            return_tensors="pt"
        )

        return {
            "pixel_values": torch.stack(pixel_values),
            "input_ids": text_batch["input_ids"],
            "attention_mask": text_batch["attention_mask"]
        }

data_collator = CustomDataCollator(tokenizer=tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir="./fine_tuned_vit_gpt2",    
    evaluation_strategy="epoch",
    learning_rate=5e-5,                   
    per_device_train_batch_size=4,        
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_strategy="epoch",                
    num_train_epochs=5,                   
    logging_dir="./logs",
    logging_steps=50,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Jalankan training untuk fine tuning ViT-GPT-2 nya pakai Flickr8k
trainer.train()

# Simpan model ViT-GPT-2 yang udah di-fine-tune ke komputer (Untuk kasus ini pake path laptopku, tapi bisa diganti dengan path yang lain)
model.save_pretrained(r"C:\Users\FELIX\Downloads\archive")
tokenizer.save_pretrained(r"C:\Users\FELIX\Downloads\archive")
