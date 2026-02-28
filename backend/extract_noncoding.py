# STEP 1: Read genome sequence
genome_seq = ""

with open("genome.txt", "r") as f:
    genome_seq = f.read().strip()

# STEP 2: Get gene positions from GFF
gene_positions = []

with open("sequence.gff3", "r") as file:
    for line in file:
        if line.startswith("#"):
            continue

        parts = line.strip().split("\t")

        if len(parts) < 5:
            continue

        if parts[2] == "gene":
            start = int(parts[3])
            end = int(parts[4])
            gene_positions.append((start, end))

# Sort genes
gene_positions.sort()

# STEP 3: Find non-coding regions
noncoding_regions = []

for i in range(len(gene_positions) - 1):
    prev_end = gene_positions[i][1]
    next_start = gene_positions[i+1][0]

    if next_start - prev_end > 50:
        noncoding_regions.append((prev_end + 1, next_start - 1))

# STEP 4: Extract sequences and save to file
with open("noncoding.fasta", "w") as out:
    count = 1
    for start, end in noncoding_regions:
        sequence = genome_seq[start-1:end]   # 0-based indexing correction
        
        out.write(f">noncoding_{count}_{start}_{end}\n")
        out.write(sequence + "\n")
        
        count += 1

print("Non-coding DNA sequences extracted successfully!")