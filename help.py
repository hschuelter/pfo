mapping = {
  # Hidrophobic
  "A": "H",
  "F": "H",
  "G": "H",
  "I": "H",
  "L": "H",
  "M": "H",
  "P": "H",
  "V": "H",
  "W": "H",
  # Polar
  "C": "P",
  "D": "P",
  "E": "P",
  "H": "P",
  "K": "P",
  "N": "P",
  "Q": "P",
  "R": "P",
  "S": "P",
  "T": "P",
  "Y": "P",
}

seq= [
  'YGGFM',
  "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN",
  "RPRTAFSSEQLARLKREFNENRYLTERRRQQLSSELGLNEAQIKIWFQNKRAKI"
]

for s in seq:
  value = ''
  for ch in s:
    value += mapping[ch]
  print(f"({len(value)}) {value}")