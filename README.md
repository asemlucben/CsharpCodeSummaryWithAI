# C# Code Summary Generator with AI

A Python tool that automatically generates documentation comments for C# methods using the Qwen2.5-1.5B-Instruct AI model.

> [!IMPORTANT]
> This project was done solely to save me a little of time and effort in writing documentation comments for C# methods. It is not intended to replace human developers or to be used in production code without thorough review.
> Always review and test the generated documentation before using it in your projects.
> The AI-generated comments may not always be accurate or complete, and should be treated as a starting point for further refinement.
> The project is not affiliated with or endorsed by any specific AI model or library, and is intended for educational and research purposes only.
> The AI model used in this project is a large language model that has been trained on a diverse range of text, but it may not always produce the desired results. Use at your own risk.

## Overview

This tool scans C# files in a specified directory, extracts method definitions, and uses an AI model to generate comprehensive XML documentation comments that follow C# documentation standards. The generated summaries include method descriptions, parameter details, and return value information.

## Features

- Recursively processes all .cs files in a specified directory
- Skips files that already have summary comments
- Identifies and extracts method declarations and bodies
- Generates XML documentation comments in standard C# format
- Supports CUDA acceleration for faster processing when available

## Requirements

- Python 3.x
- PyTorch
- Transformers library
- CUDA-capable GPU (optional, for better performance)

## Installation

```bash
# Clone this repository
git clone https://github.com/asemlucben/CsharpCodeSummaryWithAI.git
cd CsharpCodeSummaryWithAI

# Install PyTorch with CUDA support (if available)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install transformers
```

## Usage

1. Edit the `head_folder` variable in `main.py` to point to your C# project directory:

```python
head_folder = "path/to/your/csharp/project"
```

2. Run the script:

```bash
python main.py
```

The script will:
1. Check for CUDA availability
2. Find all .cs files in the specified directory
3. Extract methods from each file
4. Generate documentation comments for each method
5. Update the source files with the new documentation

## How It Works

1. The tool uses regex patterns to identify and extract C# method declarations and bodies
2. It filters out non-method code blocks like control structures, properties, and constructors
3. Each method is passed to the AI model with example inputs and outputs for proper formatting
4. The generated documentation is inserted above the method declaration in the original file

## Example

Original code:

```csharp
private int CalculateSum(int firstNumber, int secondNumber)
{
     return firstNumber + secondNumber;
}
```

After processing:

```csharp
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
private int CalculateSum(int firstNumber, int secondNumber)
{
     return firstNumber + secondNumber;
}
```

## License

MIT License