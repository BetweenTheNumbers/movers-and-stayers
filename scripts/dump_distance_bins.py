"""
Dump the birth-to-HS distance distribution in presentation-friendly bins, so the
deck histogram shows the REAL shape. Prints counts the chart will use.

Run:  python scripts/dump_distance_bins.py
"""
import pandas as pd

df = pd.read_csv('v3_analysis_with_war.csv', low_memory=False)
d = df.loc[df['is_hs_draftee'] == 1, 'distance_miles'].dropna()
d = d[d >= 0]

print(f"n with distance: {len(d):,}")
print(f"median: {d.median():.1f} mi   mean: {d.mean():.1f} mi")
print(f"share <=1mi (stayers): {(d<=1).mean()*100:.1f}%")
print(f"share >500mi (long):   {(d>500).mean()*100:.1f}%\n")

# compact bins for a histogram with a readable long tail
bins =   [0, 1, 10, 25, 50, 100, 250, 500, 1000, 2500, 100000]
labels = ["0-1","1-10","10-25","25-50","50-100","100-250",
          "250-500","500-1k","1k-2.5k","2.5k+"]
cats = pd.cut(d, bins=bins, labels=labels, right=True, include_lowest=True)
counts = cats.value_counts().reindex(labels)

print("BIN            count     share")
for lab in labels:
    c = int(counts[lab])
    print(f"  {lab:<10} {c:>7,}   {c/len(d)*100:>5.1f}%")

print("\nPaste this whole block back to drop the real shape into the slide.")
