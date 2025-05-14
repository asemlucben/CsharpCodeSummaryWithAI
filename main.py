#pip install torch --index-url https://download.pytorch.org/whl/cu118

import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

methods_list: list[str] = []

class CsharpMethod:
    declaration: str
    body: str
    def __init__(self, declaration: str, body: str):
        self.declaration = declaration.replace("[ExportMethod]", "").strip().split("\n")[0].strip()
        self.body = body

def list_torch_devices():
    """Lists all available PyTorch devices."""
    if torch.cuda.is_available():
        print("CUDA is available.")
        device_count = torch.cuda.device_count()
        print(f"Number of CUDA devices: {device_count}")
        for i in range(device_count):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available. Using CPU.")

def generate_comment(csharp_code: str = None) -> str:
    """
    Generates a comment for the provided C# code using a pre-trained model.
    Args:
        csharp_code (str): The C# code for which to generate a comment.
    """

    # Check if CUDA is available and set the device accordingly
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=device  # Use the determined device
    ).to(device)  # Move the model to the device

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    example = """
        /// <summary>
        /// This method takes two integer values and returns the sum.
        /// <example>
        /// For example:
        /// <code>
        /// int result = CalculateSum(5, 10);
        /// </code>
        /// results in <c>result</c>'s having the value 15.
        /// </example>
        /// </summary>
        /// <param name="firstNumber">The first number to sum.</param>
        /// <param name="secondNumber">The second number to sum.</param>
        /// <returns>
        /// An integer value being the sum of the input values.
        /// </returns>
    """

    prompt = "Generate the summary of the following method in C#: " \

    messages = [
        {"role": "system", "content": f"You are Luke, Your job is to create summaries of C# methods, here is an example of the output you must use: {example}. Do not generate any code, only return the summary in the expected format."},
        {"role": "user", "content": f"{prompt} ```{csharp_code}```"},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)  # Move inputs to the device

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # if the last line of the response is "///</summary>", remove it
    if response.endswith("/// </summary>"):
        response = response[:-len("/// </summary>")].strip()

    return response

def find_cs_files(root_folder):
    """
    Recursively finds all .cs files in the given folder and its subfolders.
    
    Args:
        root_folder (str): The root directory to start searching from.
        
    Returns:
        list: A list of paths to all .cs files found.
    """
    cs_files = []
    
    # Walk through all directories and files
    for root, dirs, files in os.walk(root_folder):
        # Filter only .cs files
        for file in files:
            if file.endswith('.cs'):
                # Get the full path
                file_path = os.path.join(root, file)
                cs_files.append(file_path)
    
    return cs_files

def extract_methods_from_file(file_path) -> list[CsharpMethod]:
    """
    Extracts methods from a C# file, capturing both declaration and body separately.
    
    Args:
        file_path (str): The path to the C# file.
        
    Returns:
        list[CsharpMethod]: A list of CsharpMethod objects containing declaration and body.
    """
    methods = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        if "<summary>" in content:
            # Skip files that already contain summary comments
            print(f"[-] Skipping file {file_path} as it already contains summary comments.")
            return methods

        # Improved regex that specifically targets method declarations
        # Starts with access modifiers and other method attributes
        # Ensures method name follows a return type
        # Prevents matching control structures like "if", "while", etc.
        method_pattern = r'((?:public|private|protected|internal|static|virtual|override|abstract|async|extern|\s)+\s+[\w\<\>\[\],\s\.]+\s+([A-Z][\w]+|\w+)\s*\([^)]*\)(?:\s*:\s*[^{]+)?\s*)({[^{}]*(?:{[^{}]*(?:{[^{}]*}[^{}]*)*}[^{}]*)*})'
        
        for match in re.finditer(method_pattern, content, re.DOTALL):
            declaration = match.group(1).strip()
            body = match.group(0).strip()
            
            # Additional filter to exclude control statements and constructors
            if not any(keyword in declaration.split()[0].lower() for keyword in ["catch", "if", "for", "while", "foreach", "switch", "using"]):
                # Skip properties, constructors and other non-methods
                if not re.match(r'.*\s+get\s+[{]|.*\s+set\s+[{]|.*\boperator\b|.*\bnew\b', declaration):
                    # Skip if the method is a default NetLogic method
                    if not "Start()" in declaration and not "Stop()" in declaration:
                        methods.append(CsharpMethod(declaration=declaration, body=body))
    
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
    
    return methods

def update_method_declaration(file_path: str, original_declaration: str, new_declaration: str):
    """
    Updates the method declaration with a new one.
    
    Args:
        file_path (str): The path to the C# file.
        original_declaration (str): The original method declaration to be replaced.
        new_declaration (str): The new method declaration to replace with.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        # Replace the original declaration with the new one
        updated_content = content.replace(original_declaration, new_declaration)
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(updated_content)
    except Exception as e:
        print(f"Error updating file {file_path}: {e}")    

if __name__ == "__main__":
    # Check for CUDA
    list_torch_devices()

    # Get all .cs files in the current directory
    head_folder = "D:\\dev\\Vista\\src\\IDEFiles\\Libraries"
    cs_files = find_cs_files(head_folder)
    print(f"[-] Found {len(cs_files)} csharp files in {head_folder}.")

    # Extract methods from each file and generate comments
    for file_path in cs_files:
        # Extract methods from the file
        print(f"[+] Processing file: {file_path}")
        methods = extract_methods_from_file(file_path)
        print(f"[-] Found {len(methods)} methods in {file_path}.")
        for method in methods:
            # Generate a comment for each method
            print(f" [+] Generating comment for method: {method.declaration}")
            summary = generate_comment(method.body)
            # Replace the method declaration with the generated comment
            new_declaration = summary + "\n" + method.declaration
            update_method_declaration(file_path, method.declaration, new_declaration)
            print(f" [+] Method declaration updated in {file_path}.")
        print(f"[+] Finished processing file: {file_path}")