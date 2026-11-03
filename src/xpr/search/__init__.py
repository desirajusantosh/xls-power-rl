"""Worst-case (power-virus) activity search: shared space, scorer and methods.

Every method searches the same space (a period-P pattern of full input-port
vectors {rst, vld, data}, repeated over the window) with the same evaluation
budget, scored by the same fail-closed scorer.
"""
