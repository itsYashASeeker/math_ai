from pix2text import Pix2Text
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import re
import os
import cv2
import gradio as gr 
from PIL import Image 
import numpy as np

# Initialize Pix2Text model
p2t = Pix2Text.from_config()

def do_ocr_image(image_p, task):
    print(f"Task selected: {task}")
    
    def preprocess_image(image_path):
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Use adaptive thresholding to binarize the image
        adaptive_threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        cv2.imwrite(image_path, adaptive_threshold)
        return image_path

    
    # Handle different OCR tasks based on dropdown selection
    if task == "MATH AI OCR":
        equations = []
        file_path = "."
        
        print("================================================")
        print("DOING MATH OCR ON IMAGE...\n")
        
        # Load and preprocess the image
        image = Image.open(image_p)
        image_path = os.getcwd() + "/upload/" + "sample.png"
        image.save(image_path)
        
        # Perform OCR to extract math equation text from the image
        ocr_text = p2t.recognize_formula(preprocess_image(image_path))
        equations.append(ocr_text)
        
        print("================================================")
        print("OCR IS SUCCESSFULLY DONE. PROCESSING THE EQUATIONS....\n")
        
        def correct_ocr_errors(eq):
            eq = re.sub(r'(\d)\s+(\d)', r'\1\2', eq)  # Remove spaces between digits
            eq = re.sub(r'\\frac\{(.+?)\}\{(.+?)\}', r'(\1)/(\2)', eq)  # Correct fractions
            eq = re.sub(r'(\d)(\()', r'\1*\2', eq)  # Correct bracket multiplication
            eq = re.sub(r'\\div', r'/', eq)  # Handle division symbols
            eq = eq.replace(r"{", "(").replace(r"}", ")").replace(' ', '')
            return eq

        
        def process_equation(eq_str):
            eq_str = correct_ocr_errors(eq_str)
            eq_str = eq_str.replace("^{", "**(").replace("}", ")")
            eq_str = eq_str.replace(r"\pi", "pi").replace(r"\times", "*").replace("^", "**")
            
            transformations = (standard_transformations + (implicit_multiplication_application,))
            
            if '=' in eq_str:
                lhs, rhs = eq_str.split('=', 1)
                try:
                    lhs_parsed = parse_expr(lhs, transformations=transformations, evaluate=False)
                    rhs_parsed = parse_expr(rhs, transformations=transformations, evaluate=False)
                    equation = sp.Eq(lhs_parsed, rhs_parsed)
                    return equation
                except Exception as e:
                    raise ValueError(f"Error parsing equation parts: {e}")
            else:
                try:
                    expr = parse_expr(eq_str, transformations=transformations, evaluate=False)
                    return expr
                except Exception as e:
                    raise ValueError(f"Error parsing expression: {e}")
        
        def solve_or_evaluate(eq):
            if isinstance(eq, sp.Equality):
                symbols = eq.free_symbols
                if len(symbols) == 0:
                    return "True" if eq.lhs == eq.rhs else "False"
                elif len(symbols) == 1:
                    symbol = symbols.pop()
                    solution = sp.solve(eq, symbol)
                    return solution
                else:
                    solution = sp.solve(eq)
                    return solution
            else:
                try:
                    evaluated = eq.evalf()
                    return evaluated
                except Exception as e:
                    return f"Cannot evaluate expression: {e}"
        
        result = []
        ocr_texts = []
        for eq_str in equations:
            try:
                processed_eq = process_equation(eq_str)
                result.append(solve_or_evaluate(processed_eq))  # Add the result
                ocr_texts.append(eq_str)  # Add the original OCR text
            except Exception as e:
                print(f"\nError processing equation '{eq_str}': {e}")
        
        print("PROCESSING SUCCESSFULLY DONE!!!\n")
        return ocr_texts, result  # Return both OCR text and result
    
    elif task == "IMAGE TO TEXT OCR":
        print("================================================")
        print("DOING GENERAL TEXT OCR ON IMAGE...\n")
        
        # Perform general OCR (not just math, but any text from the image)
        ocr_text = p2t.recognize_text(image_p)
        
        print("TEXT OCR IS SUCCESSFULLY DONE!\n")
        return ocr_text, "N/A"  # Returning 'N/A' as there's no evaluation to perform

# Gradio interface with dropdown
iface = gr.Interface(
    fn=do_ocr_image,
    inputs=[
        gr.Image(type="filepath"),  # Image input
        gr.Dropdown(label="Select OCR Type", choices=["MATH AI OCR", "IMAGE TO TEXT OCR"], value="MATH AI OCR")  # Use 'value' instead of 'default'
    ],
    outputs=[gr.Textbox(label="Extracted Text or Equation"), gr.Textbox(label="Result (if applicable)")],
    title="MATH AI & General OCR",
    description="Upload an image and select the OCR type (Math OCR or General Text OCR) to get the result."
)


# Launch the Gradio interface
iface.launch()
