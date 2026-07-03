# Claim Analyzer Substring Match

## Error

The claim analyzer detected `prove` inside the word `improvement`, which incorrectly marked a medium-risk claim as high risk.

## Cause

Signal matching used plain substring checks, so partial word matches were treated as real claim signals.

## Fix

Updated the claim analyzer to use word and phrase boundaries when detecting claim signals.

## Verification

Checked that:

- `significant improvement` is still detected as a medium-risk claim.
- `prove` is not detected inside `improvement`.

