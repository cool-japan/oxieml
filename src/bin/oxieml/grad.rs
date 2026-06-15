use oxieml::parser::parse;

pub(super) fn run_grad(expr: &str, wrt: usize, vars: &[f64]) {
    let tree = match parse(expr) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Parse error: {e}");
            std::process::exit(1);
        }
    };
    let lowered = tree.lower().simplify();
    let grad = lowered.grad(wrt);
    println!("Expression:    {lowered}");
    println!("d/dx{wrt}:         {grad}");

    // Optional numerical evaluation at provided variable bindings.
    if !vars.is_empty() {
        let ops = grad.to_oxiblas_ops();
        let result = oxieml::LoweredOp::eval_ops(&ops, vars);
        print!("At [");
        for (i, v) in vars.iter().enumerate() {
            if i > 0 {
                print!(", ");
            }
            print!("x{i}={v}");
        }
        println!("]:   {result}");
    }
}
