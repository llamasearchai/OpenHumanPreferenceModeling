import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import * as React from 'react';
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
  SelectGroup,
  SelectLabel,
  SelectSeparator,
} from './select';

// Radix UI Select requires standard pointers usually, 
// using generic testing-library methods.

describe('Select', () => {
  it('renders select trigger with placeholder', () => {
    render(
      <Select>
        <SelectTrigger>
          <SelectValue placeholder="Select an item" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="item1">Item 1</SelectItem>
        </SelectContent>
      </Select>
    );
    expect(screen.getByText('Select an item')).toBeInTheDocument();
  });

  it('selects an item', async () => {
    // We need to wrap in onValueChange to verify selection if controlled, 
    // or just render uncontrolled.
    const onValueChange = vi.fn();
    render(
      <Select onValueChange={onValueChange}>
        <SelectTrigger aria-label="Select">
          <SelectValue placeholder="Select" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="apple">Apple</SelectItem>
          <SelectItem value="banana">Banana</SelectItem>
        </SelectContent>
      </Select>
    );

    // Open
    const trigger = screen.getByLabelText('Select');
    fireEvent.pointerDown(trigger, { button: 0, ctrlKey: false });
    // pointerDown is often required for Radix

    // Check content open
    const apple = await screen.findByText('Apple');
    expect(apple).toBeInTheDocument();
    
    // Select
    fireEvent.click(apple);
    
    // Verify
    expect(onValueChange).toHaveBeenCalledWith('apple');
    expect(screen.getByText('Apple')).toBeInTheDocument();
  });

  it('renders label and separator', async () => {
    render(
      <Select defaultValue="apple">
        <SelectTrigger>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectGroup>
            <SelectLabel>Fruits</SelectLabel>
            <SelectItem value="apple">Apple</SelectItem>
            <SelectSeparator data-testid="select-separator" />
            <SelectItem value="carrot">Carrot</SelectItem>
          </SelectGroup>
        </SelectContent>
      </Select>
    );
    
    // Open to see items
    // Radix Primitives often need pointer events or valid DOM insertion
    // Testing library `userEvent` is better but fireEvent works if precise.
    const trigger = screen.getByRole('combobox');
    fireEvent.pointerDown(trigger);

    expect(await screen.findByText('Fruits')).toBeInTheDocument();
    // Use data-testid as separator role might be absent or implicit
    const separator = screen.getByTestId('select-separator');
    expect(separator).toBeInTheDocument();
  });
});
