from pix2text import Pix2Text
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import re
import os
import cv2
import gradio as gr 
from PIL import Image 
import PIL 
import numpy as np 




images=['math.png','math-2.png','math-3.png','math-4.png','math-5.jpg','math-6.png']



p2t = Pix2Text.from_config()


def do_ocr_image(image_p):
    print(image_p)
    def preprocess_image(image_path):
        image=cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        cv2.imwrite(image_path,blurred)
        return image_path
    
    equations=[]
    file_path="."

    print("================================================")
    print("DOING OCR ON IMAGES...\n")
    # for image in images:
    #     image_path=os.getcwd()+"/images/"+image
    #     equations.append(p2t.recognize_formula(preprocess_image(image_path)))
    image=Image.open(image_p)
    
    image_path=os.getcwd()+"/upload/"+"sample.png"
    image.save(image_path)

    equations.append(p2t.recognize_formula(preprocess_image(image_path)))

    print("================================================")
    print("OCR IS SUCCESSFULLY DONE. PROCESSING THE EQUATIONS....\n")

    def correct_ocr_errors(eq):
        eq = re.sub(r'(\d)\s+(\d)', r'\1\2', eq)
        # eq = eq.replace('l', '1').replace('O', '0')
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

    result=[]
    for eq_str in equations:
        try:
            processed_eq = process_equation(eq_str)
            # print(f"\nOriginal Equation: {eq_str}")
            # print(f"Processed Equation: {processed_eq}")
            result.append(solve_or_evaluate(processed_eq))
        except Exception as e:
            print(f"\nError processing equation '{eq_str}': {e}")
    
    print("PROCESSING SUCCESSFULLY DONE!!!\n")
    return result




# Gradio interface
iface = gr.Interface(
    fn=do_ocr_image,
    inputs=gr.Image(type="filepath"),
    outputs=[gr.Text()],
    title="MATH AI",
    description="Upload an image containing a mathematical equation to get the result.",
)

# Launch the Gradio interface
iface.launch()