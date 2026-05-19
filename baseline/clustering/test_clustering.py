from crypto.ckks import build_context, generate_keys
from clustering.clustering_optimization import ClusteringOptimizer

names_B = [
    "jack smith",
    "john smith",
    "jack smyth",
    "jane doe",
    "alice brown",
    "bob johnson",
    "jon smith",
    "jackson smith",
]

ctx = build_context()
ctx, sk = generate_keys(ctx)

optimizer = ClusteringOptimizer(k=3)
optimizer.fit(names_B)

plain_result = optimizer.search_plain("jack smith")
enc_result = optimizer.encrypted_search("jack smith", ctx, sk)

print("plain:", plain_result["catch"], plain_result["matched_names"])
print("enc:", enc_result["catch"], enc_result["matched_names"])
print("enc scores:", enc_result["decrypted_column_scores"])
