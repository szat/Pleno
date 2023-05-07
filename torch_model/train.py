import sys
sys.path.append('.')

from options import RadianceFieldOptions
from trainer import Trainer


options = RadianceFieldOptions()
opts = options.parse()

if __name__=="__main__":
    trainer = Trainer(opts)
    trainer.train()
