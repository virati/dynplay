# %%
from dysts.flows import Lorenz

model = Lorenz()
sol = model.make_trajectory(1000)  # (1000, 3)

# %%
import matplotlib.pyplot as plt

plt.plot(sol[:, 0], sol[:, 1])

# %%
# We want to talk to our dynamical system now
import dspy
import dspy
from dotenv import load_dotenv
import os
from typing import List
import numpy as np

load_dotenv()
dspy.configure(
    lm=dspy.LM(
        "gemini/gemini-2.5-flash",
        api_key=os.getenv("GEMINI_KEY"),
        temperature=1.0,
        max_tokens=6000,
    )
)

# %%
from pydantic import BaseModel, Field
import numpy as np


# Define the Pydantic model for your array
class NumpyArrayField(BaseModel):
    # Use Field() to give a description to the LLM
    data: np.ndarray = Field(description="A NumPy array containing numerical data.")

    class Config:
        arbitrary_types_allowed = True


# %
class Dynsight(dspy.Signature):
    trajectory: NumpyArrayField = dspy.InputField(
        description="The trajectory of the dynamical system."
    )
    equation: str = dspy.InputField(description="The equation of the dynamical system.")
    insight: str = dspy.OutputField(
        description="The insight about the dynamical system."
    )


# %%
summarizer = dspy.ChainOfThought(Dynsight)
results = summarizer(trajectory=sol, equation=model.description)
# %%
print(results.insight)
# %%
# Now, can we talk with the results?
