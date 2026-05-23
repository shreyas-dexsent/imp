export function html(strings, ...values) {
  return strings.reduce((result, part, index) => `${result}${part}${values[index] ?? ''}`, '');
}

export function qs(root, selector) {
  const element = root.querySelector(selector);
  if (!element) throw new Error(`Missing UI element: ${selector}`);
  return element;
}
