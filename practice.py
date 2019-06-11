from jitcode import jitcode, y
import pdb

substances = Ca, Na, Ko = [y(i) for i in range(3)]
diffusion_rate = { Ca: 0.3, Na: 0.4, Ko: 0.5 }
def diffusion_loss(substance):
  return -diffusion_rate[substance]*substance
pdb.set_trace()
f = {substance: diffusion_loss(substance) for substance in substances}
ODE = jitcode(f)