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
        self.declaration = declaration.strip().split("\n")[0].strip()
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

def validate_summary(summary: str) -> bool:
    """
    Validates the generated summary to ensure it contains the expected format.
    
    Args:
        summary (str): The generated summary to validate.
        
    Returns:
        bool: True if the summary is valid, False otherwise.
    """
    error_message = ""
    valid_summary = True

    # Check if the summary contains the expected tags
    if not "<summary>" in summary or not "</summary>" in summary:
        error_message = "[!] Summary does not contain <summary> or </summary> tags.\n"
        valid_summary = False
    if "<code>" in summary and "</code>" not in summary:
        error_message = "[!] Summary contains <code> tag without closing </code> tag.\n"
        valid_summary = False
    if "<example>" in summary and "</example>" not in summary:
        error_message = "[!] Summary contains <example> tag without closing </example> tag.\n"
        valid_summary = False
    if "<returns>" in summary and "</returns>" not in summary:
        error_message = "[!] Summary contains <returns> tag without closing </returns> tag.\n"
        valid_summary = False
    if "<param" in summary and "</param>" not in summary:
        error_message = "[!] Summary contains <param> tag without closing </param> tag.\n"
        valid_summary = False
    if "<exception" in summary and "</exception>" not in summary:
        error_message = "[!] Summary contains <exception> tag without closing </exception> tag.\n"
        valid_summary = False
    if "<example>" in summary and "</example>" not in summary:
        error_message = "[!] Summary contains <example> tag without closing </example> tag.\n"
        valid_summary = False
    
    # Count if the number of opened <param> tags is equal to the number of closed </param> tags
    opened_tags = re.findall(r'<[a-zA-Z]+.*>', summary)
    # Remove the inline tags from the opened tags
    opened_tags = [tag for tag in opened_tags if not re.search(r'<[a-zA-Z]+.*?/>', tag)]
    # Count the number of closed </param> tags
    closed_tags = re.findall(r'</[a-zA-Z]+>', summary)
    if len(opened_tags) != len(closed_tags):
        error_message = "[!] Summary contains an unequal number of opened and closed tags.\n"
        valid_summary = False

    # Count if too many <summary> tags are present
    summary_tags = re.findall(r'<summary>', summary)
    if len(summary_tags) > 1:
        error_message = "[!] Summary contains more than one <summary> tag.\n"
        valid_summary = False
    # Count if too many <returns> tags are present
    returns_tags = re.findall(r'<returns>', summary)
    if len(returns_tags) > 1:
        error_message = "[!] Summary contains more than one <returns> tag.\n"
        valid_summary = False
    # Count if too many <remarks> tags are present
    remarks_tags = re.findall(r'<remarks>', summary)
    if len(remarks_tags) > 1:
        error_message = "[!] Summary contains more than one <remarks> tag.\n"
        valid_summary = False

    if not valid_summary:
        try:
            with open("error_log_invalid_summaries.txt", "a", encoding="utf-8") as error_log:
                error_log.write("========================================\n")
                error_log.write(f"Error: {error_message}\n")
                error_log.write(f"Invalid summary: \n{summary}\n")
                error_log.write("========================================\n")
        except Exception as e:
            print(f"[!] Error writing to error log: {e}")
        
        print(error_message)
        return False

    return True

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

    sample_input_1 = """
        private int CalculateSum(int firstNumber, int secondNumber)
        {
            return firstNumber + secondNumber;
        }
    """
    sample_input_2 = """
        private int GetRandomValue()
        {
            return rnd.Next(0, 100) + rnd.Next(0, 100);
        }
    """
    sample_output_1 = """
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
    sample_output_2 = """
        /// <summary>
        /// This method generates a random integer value between 0 and 100.
        /// <example>
        /// For example:
        /// <code>
        /// int randomValue = GetRandomValue();
        /// </code>
        /// results in <c>randomValue</c>'s having a random value between 0 and 200.
        /// </example>
        /// </summary>
        /// <returns>
        /// An integer value being the random value generated.
        /// </returns>
    """

    prompt = "Generate the summary of the following method in C#: " \

    messages = [
        {"role": "system", "content": f"Your job is to create summaries of C# methods. For example, the following code: \"{sample_input_1}\" should return: \"{sample_output_1}\", while \"{sample_input_2}\" should return: \"{sample_output_2}\". Do not generate any code, do not include exceptions in the summary, only return the summary in the expected format."},
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

    # Remove the closing <summary> tag which is sometimes wrongly generated
    if len(re.findall(r'/// </summary>', response)) > 1 and response.endswith("/// </summary>"):
        response = response[:-len("/// </summary>")].strip()

    # Remove exception tags
    response = re.sub(r'/// <exception.*?</exception>', '', response, flags=re.DOTALL).strip()
    # Remove example tags
    response = re.sub(r'/// <example.*?</example>', '', response, flags=re.DOTALL).strip()
    # Remove blank tags
    response = re.sub(r'/// <.*></.*>', '', response, flags=re.DOTALL).strip()

    # Make sure the response only contains the summary
    response = "\n".join([line for line in response.split("\n") if line.startswith("///")])

    # For some reason, the model sometimes generates another summary after the remarks tag
    if "</remarks>" in response:
        # Remove all text after the first </remarks> tag
        response = response.split("</remarks>")[0] + "</remarks>"
        
    # Check if the summary is valid
    if not validate_summary(response):
        raise ValueError("[!] Generated summary is not valid.")

    return response.strip()

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

        content = re.sub(r'#region\s+.*?\n', '', content, flags=re.DOTALL).strip()
        content = re.sub(r'#endregion\s+.*?\n', '', content, flags=re.DOTALL).strip()
        content = re.sub(r'//.*?\n', '', content, flags=re.DOTALL).strip()
        content = content.replace("[ExportMethod]", "").strip()

        # Starts with access modifiers and other method attributes
        method_pattern = r'((?:public|private|protected|internal|static|virtual|override|abstract|async|extern|\s)+\s+[\w\<\>\[\],\s\.]+\s+([A-Z][\w]+|\w+)\s*\([^)]*\)(?:\s*:\s*[^{]+)?\s*)({[^{}]*(?:{[^{}]*(?:{[^{}]*}[^{}]*)*}[^{}]*)*})'
        
        for match in re.finditer(method_pattern, content, re.DOTALL):
            declaration = match.group(1).strip()
            body = match.group(0).strip()
            
            # Additional filter to exclude control statements and constructors
            restricted_keywords = [
                "if", "for", "while", "foreach", "switch", "using", "try", "catch",
                "new", "get ", "set ", "#region", "#endregion", "else", "return", "throw",
                "break", "continue", "goto", "default", "lock", "checked", "unchecked"
            ]

            process = True
            for keyword in restricted_keywords:
                if declaration.startswith(keyword):
                    process = False
                    break
            
            if process:
                # Skip properties, constructors and other non-methods
                if not re.match(r'.*\s+get\s+[{]|.*\s+set\s+[{]|.*\boperator\b|.*\bnew\b', declaration):
                    # Skip some specific method names
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

    # Remove the invalid summaries file if it exists
    if os.path.exists("error_log_invalid_summaries.txt"):
        os.remove("error_log_invalid_summaries.txt")
        print("[-] Removed the invalid summaries file.")

    # Get all .cs files in the current directory
    head_folder = "D:\\dev\\Vista\\src\\IDEFiles\\Libraries"
    cs_files = find_cs_files(head_folder)
    print(f"[-] Found {len(cs_files)} csharp files in {head_folder}.")

    # Extract methods from each file and generate comments
    with open("error_log.txt", "w", encoding="utf-8") as error_log:
        error_log.write("Error log for C# method comment generation:\n")
        error_log.write("========================================\n")
        error_log.write("Errors encountered during processing:\n\n")
        for file_path in cs_files:
            try:
                # Extract methods from the file
                print(f"[+] Processing file: {file_path}")
                methods_to_update = extract_methods_from_file(file_path)
                print(f"[-] Found {len(methods_to_update)} methods in {file_path}.")
                if (len(methods_to_update) == 0):
                    continue
                for method in methods_to_update:
                    try:
                        # Generate a comment for each method
                        print(f" [+] Generating comment for method: {method.declaration}")
                        summary = generate_comment(method.body)
                        # Replace the method declaration with the generated comment
                        new_body = summary + "\n" + method.body
                        update_method_declaration(file_path, method.body, new_body)
                        print(f" [+] Method declaration updated in {file_path}.")
                    except Exception as e:
                        print(f"[!] Error generating comment for method {method.declaration}: {e}")
                        error_log.write(f"Error generating comment for method {method.declaration}: {e}\n")
                        error_log.write(f"File: {file_path}\n")
                        error_log.write("========================================\n")
                print(f"[+] Finished processing file: {file_path}")
            except Exception as e:
                print(f"[!] Error processing file {file_path}: {e}")
                error_log.write(f"Error processing file {file_path}: {e}\n")
                error_log.write("========================================\n")
    print("[-] Finished processing all files.")
