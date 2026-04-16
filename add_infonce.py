import nbformat

with open('mnlp.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

already_added = any("MultipleNegativesRankingLoss" in cell.source for cell in nb.cells if cell.cell_type == 'code')

if not already_added:
    md_cell = nbformat.v4.new_markdown_cell(source="""## Alternative: Fine-Tuning with InfoNCE + Hard Negatives
Instead of replacing the Triplet Loss, we can train a separate model using `MultipleNegativesRankingLoss` (che implementa InfoNCE).
This uses the provided negative chunk as a "hard negative" and all other positive chunks in the batch as "in-batch negatives".""")

    train_code = """# Fine-tune DistilBERT with MultipleNegativesRankingLoss (InfoNCE)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers import SentenceTransformer, models as st_models

print("\\n" + "="*70)
print("PHASE 2 (Alternative): FINE-TUNING WITH INFONCE + HARD NEGATIVES")
print("="*70 + "\\n")

# Load base model again to start fresh (we don't want to fine-tune an already fine-tuned model)
model_name = "distilbert-base-uncased"
print(f"[1/3] Loading {model_name}...")

try:
    word_embedding_model = st_models.Transformer(model_name, max_seq_length=256)
    pooling_model = st_models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model_infonce = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    print(f"✓ Model loaded successfully\\n")
except Exception as e:
    model_infonce = SentenceTransformer('all-distilroberta-v1')
    print(f"✓ Loaded alternative model: all-distilroberta-v1\\n")

# Define InfoNCE loss
infonce_loss = MultipleNegativesRankingLoss(model=model_infonce)

print(f"[2/3] Configuring training...")
print(f"  - Loss function: MultipleNegativesRankingLoss (InfoNCE + Hard Negatives)")
print(f"  - Batch size: 32")
print(f"  - Epochs: 3")

epochs = 3
warmup_steps = int(len(train_dataloader) * 0.1)

print(f"[3/3] Starting fine-tuning...")
print("-" * 70)

model_infonce.fit(
    train_objectives=[(train_dataloader, infonce_loss)],
    epochs=epochs,
    warmup_steps=warmup_steps,
    show_progress_bar=True,
    checkpoint_path="./distilbert-semantic-search-infonce",
    checkpoint_save_steps=len(train_dataloader),
    checkpoint_save_total_limit=3,
)

print("\\n✓ Fine-tuning complete!")
print(f"✓ Model checkpoints saved to: ./distilbert-semantic-search-infonce")"""
    train_cell = nbformat.v4.new_code_cell(source=train_code)

    eval_code = """# Evaluate InfoNCE fine-tuned model
print("\\n" + "="*70)
print("PHASE 3 (Alternative): EVALUATION - INFONCE")
print("="*70 + "\\n")

infonce_results = evaluate_finetuned_model(model_infonce, test_data, 'test')

# Compare all results
print("\\n" + "="*85)
print("FINAL RESULTS COMPARISON: BASELINE vs TRIPLET vs INFONCE")
print("="*85)
print(f"\\n{'Metric':<10} {'Baseline':<15} {'Triplet Loss':<15} {'InfoNCE':<15} {'Best Impr.':<15}")
print("-" * 85)

for metric in ['hit@1', 'hit@3', 'hit@5']:
    baseline = model_1_results[metric]
    triplet = finetuned_results[metric]
    infonce = infonce_results[metric]
    
    best_finetuned = max(triplet, infonce)
    improvement = ((best_finetuned - baseline) / baseline) * 100
    
    print(f"{metric:<10} {baseline:<15.4f} {triplet:<15.4f} {infonce:<15.4f} {improvement:>+6.2f}%")

print("="*85)"""
    eval_cell = nbformat.v4.new_code_cell(source=eval_code)

    nb.cells.extend([md_cell, train_cell, eval_cell])

    with open('mnlp.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print("Cells appended successfully.")
else:
    print("Cells already present.")