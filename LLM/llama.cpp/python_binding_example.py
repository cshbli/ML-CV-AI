from llama_cpp import Llama

# GLOBAL VARIABLES
my_model_path = "./model/zephyr-7b-beta.Q4_0.gguf"
CONTEXT_SIZE = 512

# LOAD THE MODEL
zephyr_model = Llama(model_path=my_model_path,
                    n_ctx=CONTEXT_SIZE)

def generate_text_from_prompt(user_prompt,
                             max_tokens = 100,
                             temperature = 0.3,
                             top_p = 0.1,
                             echo = True,
                             stop = ["Q", "\n"]):
   # Define the parameters
   model_output = zephyr_model(
       user_prompt,
       max_tokens=max_tokens,
       temperature=temperature,
       top_p=top_p,
       echo=echo,
       stop=stop,
   )

   return model_output


if __name__ == "__main__":

    my_prompt = "What do you think about the inclusion policies in Tech companies?"

    zephyr_model_response = generate_text_from_prompt(my_prompt)

    print(zephyr_model_response)

    final_result = zephyr_model_response["choices"][0]["text"].strip()

    print(final_result)
