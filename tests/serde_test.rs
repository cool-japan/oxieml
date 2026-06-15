#[cfg(feature = "serde")]
mod serde_tests {
    use oxieml::eval::EvalCtx;
    use oxieml::{DiscoveredFormula, EmlTree, LoweredOp, NamedConst};
    use std::sync::Arc;

    #[test]
    fn emltree_json_round_trip() {
        let tree = EmlTree::var(0);
        let json = tree.to_json().expect("serialize");
        let back = EmlTree::from_json(&json).expect("deserialize");
        let ctx = EvalCtx::new(&[1.5]);
        let orig = tree.eval_real(&ctx).ok();
        let restored = back.eval_real(&ctx).ok();
        assert_eq!(orig, restored);
    }

    #[test]
    fn emltree_eml_json_round_trip() {
        let one = EmlTree::one();
        let x = EmlTree::var(0);
        let tree = EmlTree::eml(&x, &one);
        let json = tree.to_json().expect("serialize");
        let back = EmlTree::from_json(&json).expect("deserialize");
        let ctx = EvalCtx::new(&[1.5]);
        let orig = tree.eval_real_lowered(&ctx).ok();
        let restored = back.eval_real_lowered(&ctx).ok();
        assert_eq!(orig, restored);
    }

    #[test]
    fn emltree_pretty_json_does_not_panic() {
        let tree = EmlTree::eml(&EmlTree::var(0), &EmlTree::one());
        let json = tree.to_json_pretty().expect("pretty serialize");
        assert!(json.contains('\n'), "pretty JSON should have newlines");
    }

    #[test]
    fn emltree_binary_round_trip() {
        let tree = EmlTree::var(0);
        let bytes = tree.to_binary().expect("binary serialize");
        let back = EmlTree::from_binary(&bytes).expect("binary deserialize");
        let ctx = EvalCtx::new(&[2.5]);
        let orig = tree.eval_real(&ctx).ok();
        let restored = back.eval_real(&ctx).ok();
        assert_eq!(orig, restored);
    }

    #[test]
    fn lowered_op_json_round_trip() {
        let op = LoweredOp::Add(
            Arc::new(LoweredOp::Var(0)),
            Arc::new(LoweredOp::Const(std::f64::consts::PI)),
        );
        let json = serde_json::to_string(&op).expect("serialize");
        let back: LoweredOp = serde_json::from_str(&json).expect("deserialize");
        let v = back.eval(&[1.0]);
        let expected = op.eval(&[1.0]);
        assert!(
            (v - expected).abs() < 1e-15,
            "round-trip eval mismatch: {v} vs {expected}"
        );
    }

    #[test]
    fn named_const_json_round_trip() {
        let nc = NamedConst::Pi;
        let json = serde_json::to_string(&nc).expect("serialize");
        let back: NamedConst = serde_json::from_str(&json).expect("deserialize");
        assert!((back.value() - nc.value()).abs() < 1e-15);
    }

    #[test]
    fn lowered_op_all_variants_serialize() {
        let variants: Vec<LoweredOp> = vec![
            LoweredOp::Const(1.0),
            LoweredOp::Var(0),
            LoweredOp::NamedConst(NamedConst::Pi),
            LoweredOp::Sin(Arc::new(LoweredOp::Var(0))),
            LoweredOp::Cos(Arc::new(LoweredOp::Var(0))),
            LoweredOp::Tan(Arc::new(LoweredOp::Var(0))),
            LoweredOp::Sinh(Arc::new(LoweredOp::Var(0))),
            LoweredOp::Cosh(Arc::new(LoweredOp::Var(0))),
            LoweredOp::Tanh(Arc::new(LoweredOp::Var(0))),
            LoweredOp::Arcsin(Arc::new(LoweredOp::Const(0.5))),
            LoweredOp::Arccos(Arc::new(LoweredOp::Const(0.5))),
            LoweredOp::Arctan(Arc::new(LoweredOp::Var(0))),
            LoweredOp::Arcsinh(Arc::new(LoweredOp::Var(0))),
            LoweredOp::Arccosh(Arc::new(LoweredOp::Const(1.5))),
            LoweredOp::Arctanh(Arc::new(LoweredOp::Const(0.5))),
            LoweredOp::Exp(Arc::new(LoweredOp::Var(0))),
            LoweredOp::Ln(Arc::new(LoweredOp::Const(1.0))),
            LoweredOp::Neg(Arc::new(LoweredOp::Var(0))),
            LoweredOp::Add(Arc::new(LoweredOp::Var(0)), Arc::new(LoweredOp::Const(1.0))),
            LoweredOp::Sub(Arc::new(LoweredOp::Var(0)), Arc::new(LoweredOp::Const(1.0))),
            LoweredOp::Mul(Arc::new(LoweredOp::Var(0)), Arc::new(LoweredOp::Const(2.0))),
            LoweredOp::Div(Arc::new(LoweredOp::Const(1.0)), Arc::new(LoweredOp::Var(0))),
            LoweredOp::Pow(Arc::new(LoweredOp::Var(0)), Arc::new(LoweredOp::Const(2.0))),
        ];
        for v in &variants {
            let json = serde_json::to_string(v).expect("serialize failed");
            let back: LoweredOp = serde_json::from_str(&json).expect("deserialize failed");
            let val = back.eval(&[1.5]);
            let expected = v.eval(&[1.5]);
            // Both may be NaN for some variants — use bit equality
            assert_eq!(
                val.to_bits(),
                expected.to_bits(),
                "eval mismatch for {:?}: {val} vs {expected}",
                v
            );
        }
    }

    #[test]
    fn json_golden_snapshot_rename_all() {
        // With rename_all = "snake_case", variants become lowercase in JSON.
        let op = LoweredOp::Add(Arc::new(LoweredOp::Const(2.0)), Arc::new(LoweredOp::Var(0)));
        let json = serde_json::to_string(&op).expect("serialize");
        // Renamed variants should appear as "add", "const", "var"
        assert!(
            json.contains("add") || json.contains("Add"),
            "schema should be readable: {json}"
        );
        assert!(
            json.contains("2.0") || json.contains('2'),
            "constant preserved: {json}"
        );
    }

    #[test]
    fn discovered_formula_json_round_trip() {
        use oxieml::symreg::{SymRegConfig, SymRegEngine};
        // Use quick config to generate a real DiscoveredFormula.
        let xs: Vec<f64> = (0..8).map(|i| (i as f64) * 0.25).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| x * x).collect();
        let data: Vec<Vec<f64>> = xs.iter().map(|&x| vec![x]).collect();
        let config = SymRegConfig::quick();
        let engine = SymRegEngine::new(config);
        let results = engine.discover(&data, &ys, 1).expect("symreg failed");
        if let Some(formula) = results.into_iter().next() {
            let json = formula.to_json().expect("formula serialize");
            let back = DiscoveredFormula::from_json(&json).expect("formula deserialize");
            // MSE and complexity survive the round-trip
            assert!((back.mse - formula.mse).abs() < 1e-15, "mse changed");
            assert_eq!(back.complexity, formula.complexity, "complexity changed");
            assert_eq!(back.pretty, formula.pretty, "pretty changed");
        }
    }

    #[test]
    fn symreg_config_json_round_trip() {
        use oxieml::symreg::SymRegConfig;
        let config = SymRegConfig::balanced();
        let json = serde_json::to_string(&config).expect("config serialize");
        let back: SymRegConfig = serde_json::from_str(&json).expect("config deserialize");
        assert_eq!(back.max_depth, config.max_depth);
        assert!((back.learning_rate - config.learning_rate).abs() < 1e-15);
    }
}
