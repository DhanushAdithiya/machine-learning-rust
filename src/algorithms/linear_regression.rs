struct InputNumbers<T> {
    x_values: Vec<T>,
    y_values: Vec<T>,
}

impl<T> InputNumbers<T>
where
    T: std::ops::Add<Output = T>
        + std::clone::Clone
        + std::default::Default
        + std::ops::Div<Output = T>
        + Into<f64>
        + Copy,
{
    fn new(x_values: Vec<T>, y_values: Vec<T>) -> InputNumbers<T> {
        InputNumbers { x_values, y_values }
    }

    fn find_mean(&self) -> Option<(f64, f64)> {
        let sum_x: T = self
            .x_values
            .iter()
            .cloned()
            .fold(T::default(), |acc, x| acc + x);
        let count_x = self.x_values.len() as u32;

        let sum_y: T = self
            .y_values
            .iter()
            .cloned()
            .fold(T::default(), |acc, x| acc + x);
        let count_y = self.y_values.len() as u32;

        if count_x > 0 {
            Some((
                sum_x.into() / (count_x as f64),
                sum_y.into() / (count_y as f64),
            ))
        } else {
            None
        }
    }

    pub fn find_intercept_and_slope(&self) -> Option<(f64, f64)> {
        let value_count = self.x_values.len();
        let (x_mean, y_mean) = self.find_mean().unwrap();
        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for i in 0..value_count {
            numerator += (self.x_values[i].into() - x_mean) * (self.y_values[i].into() - y_mean);
            denominator += (self.x_values[i].into() - x_mean).powf(2.0);
        }

        let gradient = numerator / denominator;
        let intercept = y_mean - (gradient * x_mean);

        return Some((intercept, gradient));
    }

    fn calculate_score(&self) -> Option<f64> {
        let mut ssr_t = 0.0;
        let mut ssr_r = 0.0;
        let (_, y_mean) = self.find_mean().unwrap();
        let (intercept, gradient) = self.find_intercept_and_slope().unwrap();
        for i in 0..self.x_values.len() {
            let y_pred = intercept + (gradient * self.x_values[i].into());
            ssr_t += (self.y_values[i].into() - y_mean).powf(2.0);
            ssr_r += (self.y_values[i].into() - y_pred).powf(2.0);
        }

        let r2 = 1.0 - (ssr_r / ssr_t);
        return Some(r2);
    }

    fn test_model(&self, intercept: f64, gradient: f64) -> Option<Vec<f64>> {
        let mut y_pred: Vec<f64> = Vec::new();
        for i in 0..self.x_values.len() {
            let predicted_val = intercept + (gradient * self.x_values[i].into());
            y_pred.push(predicted_val);
        }
        Some(y_pred)
    }

    fn rmse(&self, pred_values: Vec<f64>) -> Option<f64> {
        let mut total = 0.0;
        for i in 0..self.x_values.len() {
            total += (pred_values[i] - self.y_values[i].into()).powf(2.0);
        }

        return Some((total / self.x_values.len() as f64).sqrt());
    }
}

pub fn test() {
    let ind_values = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let dep_values = vec![1, 4, 9, 16, 25, 36, 49, 64];

    let test_ind = vec![10];
    let test_dep = vec![100];

    let calc = InputNumbers::new(ind_values, dep_values);
    let test = InputNumbers::new(test_ind, test_dep);

    let (intercept, gradient) = calc.find_intercept_and_slope().unwrap();
    println!("{},{}", intercept, gradient);
    let preditions = test.test_model(intercept, gradient).unwrap();
    println!("ACCURACY: {}", test.rmse(preditions).unwrap());

    println!("{:?}", calc.calculate_score().unwrap() * 100.0);
}
