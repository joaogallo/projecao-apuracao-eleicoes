"""Projeção de resultado eleitoral durante a apuração.

O modelo segue a lógica descrita por Rodrigo Mahlmeister (NeoFeed, out/2026):
em vez de estimar intenção de voto, parte dos votos já apurados, mede o *swing*
de cada local de votação em relação à eleição de referência (2022) e projeta
as urnas pendentes com uma regressão hierárquica (nacional -> região -> UF)
que considera o perfil de votação e o porte do município.
"""
