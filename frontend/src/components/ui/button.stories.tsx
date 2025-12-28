/**
 * Button Component Stories
 * 
 * Purpose: Storybook documentation for Button component
 * Showcases all variants, sizes, states, and use cases
 */

import type { Meta, StoryObj } from '@storybook/react';
import { Button } from './button';
import { Check, ChevronRight, Download, Mail, Plus, Trash2, X } from 'lucide-react';

const meta: Meta<typeof Button> = {
  title: 'UI/Button',
  component: Button,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: `
A versatile button component that supports multiple variants, sizes, and states.
Built with Radix UI primitives for optimal accessibility.

## Features
- 6 visual variants: default, destructive, outline, secondary, ghost, link
- 4 sizes: default, small, large, icon
- Loading state with spinner
- Left and right icon support
- Full width option
- Polymorphic with \`asChild\` prop
        `,
      },
    },
  },
  tags: ['autodocs'],
  argTypes: {
    variant: {
      control: 'select',
      options: ['default', 'destructive', 'outline', 'secondary', 'ghost', 'link'],
      description: 'The visual style variant of the button',
    },
    size: {
      control: 'select',
      options: ['default', 'sm', 'lg', 'icon'],
      description: 'The size of the button',
    },
    isLoading: {
      control: 'boolean',
      description: 'Shows a loading spinner and disables the button',
    },
    disabled: {
      control: 'boolean',
      description: 'Disables the button',
    },
    fullWidth: {
      control: 'boolean',
      description: 'Makes the button full width of its container',
    },
    asChild: {
      control: 'boolean',
      description: 'Renders the button as its child element',
    },
    children: {
      control: 'text',
      description: 'Button content',
    },
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

// ===========================================================================
// Base Stories
// ===========================================================================

export const Default: Story = {
  args: {
    children: 'Button',
  },
};

export const WithOnClick: Story = {
  args: {
    children: 'Click me',
    onClick: () => alert('Button clicked!'),
  },
};

// ===========================================================================
// Variant Stories
// ===========================================================================

export const Variants: Story = {
  render: () => (
    <div className="flex flex-wrap gap-4">
      <Button variant="default">Default</Button>
      <Button variant="destructive">Destructive</Button>
      <Button variant="outline">Outline</Button>
      <Button variant="secondary">Secondary</Button>
      <Button variant="ghost">Ghost</Button>
      <Button variant="link">Link</Button>
    </div>
  ),
};

export const Destructive: Story = {
  args: {
    variant: 'destructive',
    children: 'Delete',
  },
};

export const Outline: Story = {
  args: {
    variant: 'outline',
    children: 'Outline',
  },
};

export const Secondary: Story = {
  args: {
    variant: 'secondary',
    children: 'Secondary',
  },
};

export const Ghost: Story = {
  args: {
    variant: 'ghost',
    children: 'Ghost',
  },
};

export const Link: Story = {
  args: {
    variant: 'link',
    children: 'Link',
  },
};

// ===========================================================================
// Size Stories
// ===========================================================================

export const Sizes: Story = {
  render: () => (
    <div className="flex items-center gap-4">
      <Button size="sm">Small</Button>
      <Button size="default">Default</Button>
      <Button size="lg">Large</Button>
      <Button size="icon"><Plus className="h-4 w-4" /></Button>
    </div>
  ),
};

export const Small: Story = {
  args: {
    size: 'sm',
    children: 'Small',
  },
};

export const Large: Story = {
  args: {
    size: 'lg',
    children: 'Large',
  },
};

export const IconSize: Story = {
  args: {
    size: 'icon',
    children: <Plus className="h-4 w-4" />,
    'aria-label': 'Add item',
  },
};

// ===========================================================================
// Icon Stories
// ===========================================================================

export const WithLeftIcon: Story = {
  args: {
    leftIcon: <Mail className="h-4 w-4" />,
    children: 'Login with Email',
  },
};

export const WithRightIcon: Story = {
  args: {
    rightIcon: <ChevronRight className="h-4 w-4" />,
    children: 'Continue',
  },
};

export const WithBothIcons: Story = {
  args: {
    leftIcon: <Download className="h-4 w-4" />,
    rightIcon: <Check className="h-4 w-4" />,
    children: 'Download',
  },
};

export const IconButtons: Story = {
  render: () => (
    <div className="flex gap-2">
      <Button size="icon" variant="default" aria-label="Add">
        <Plus className="h-4 w-4" />
      </Button>
      <Button size="icon" variant="secondary" aria-label="Close">
        <X className="h-4 w-4" />
      </Button>
      <Button size="icon" variant="outline" aria-label="Download">
        <Download className="h-4 w-4" />
      </Button>
      <Button size="icon" variant="destructive" aria-label="Delete">
        <Trash2 className="h-4 w-4" />
      </Button>
      <Button size="icon" variant="ghost" aria-label="More">
        <ChevronRight className="h-4 w-4" />
      </Button>
    </div>
  ),
};

// ===========================================================================
// State Stories
// ===========================================================================

export const Loading: Story = {
  args: {
    isLoading: true,
    children: 'Submitting...',
  },
};

export const LoadingStates: Story = {
  render: () => (
    <div className="flex gap-4">
      <Button isLoading>Default</Button>
      <Button isLoading variant="destructive">Destructive</Button>
      <Button isLoading variant="outline">Outline</Button>
      <Button isLoading variant="secondary">Secondary</Button>
    </div>
  ),
};

export const LoadingWithIcon: Story = {
  args: {
    isLoading: true,
    leftIcon: <Mail className="h-4 w-4" />,
    children: 'Sending...',
  },
};

export const Disabled: Story = {
  args: {
    disabled: true,
    children: 'Disabled',
  },
};

export const DisabledVariants: Story = {
  render: () => (
    <div className="flex gap-4">
      <Button disabled>Default</Button>
      <Button disabled variant="destructive">Destructive</Button>
      <Button disabled variant="outline">Outline</Button>
      <Button disabled variant="secondary">Secondary</Button>
      <Button disabled variant="ghost">Ghost</Button>
      <Button disabled variant="link">Link</Button>
    </div>
  ),
};

// ===========================================================================
// Layout Stories
// ===========================================================================

export const FullWidth: Story = {
  args: {
    fullWidth: true,
    children: 'Full Width Button',
  },
  decorators: [
    (Story) => (
      <div className="w-96">
        <Story />
      </div>
    ),
  ],
};

export const FullWidthVariants: Story = {
  decorators: [
    (Story) => (
      <div className="w-96">
        <Story />
      </div>
    ),
  ],
  render: () => (
    <div className="space-y-2">
      <Button fullWidth>Default Full Width</Button>
      <Button fullWidth variant="outline">Outline Full Width</Button>
      <Button fullWidth variant="secondary">Secondary Full Width</Button>
    </div>
  ),
};

// ===========================================================================
// Polymorphic Stories
// ===========================================================================

export const AsLink: Story = {
  args: {
    asChild: true,
    children: <a href="#">Link styled as button</a>,
  },
};

export const AsExternalLink: Story = {
  args: {
    asChild: true,
    variant: 'outline',
    children: (
      <a href="https://github.com" target="_blank" rel="noopener noreferrer">
        Visit GitHub
      </a>
    ),
  },
};

// ===========================================================================
// Real-world Example Stories
// ===========================================================================

export const FormActions: Story = {
  render: () => (
    <div className="flex gap-2 justify-end border-t pt-4">
      <Button variant="ghost">Cancel</Button>
      <Button variant="outline">Save Draft</Button>
      <Button>Publish</Button>
    </div>
  ),
};

export const DestructiveConfirmation: Story = {
  render: () => (
    <div className="p-6 border rounded-lg max-w-sm">
      <h3 className="font-semibold text-lg">Delete Item</h3>
      <p className="text-sm text-muted-foreground mt-1 mb-4">
        Are you sure you want to delete this item? This action cannot be undone.
      </p>
      <div className="flex gap-2 justify-end">
        <Button variant="outline">Cancel</Button>
        <Button variant="destructive" leftIcon={<Trash2 className="h-4 w-4" />}>
          Delete
        </Button>
      </div>
    </div>
  ),
};

export const ToolbarButtons: Story = {
  render: () => (
    <div className="flex items-center gap-1 p-2 border rounded">
      <Button size="icon" variant="ghost" aria-label="Bold">
        <span className="font-bold">B</span>
      </Button>
      <Button size="icon" variant="ghost" aria-label="Italic">
        <span className="italic">I</span>
      </Button>
      <Button size="icon" variant="ghost" aria-label="Underline">
        <span className="underline">U</span>
      </Button>
      <div className="w-px h-6 bg-border mx-1" />
      <Button size="sm" variant="ghost">
        Insert Link
      </Button>
    </div>
  ),
};

export const CallToAction: Story = {
  render: () => (
    <div className="text-center space-y-4 p-8 bg-gradient-to-br from-primary/10 to-primary/5 rounded-xl">
      <h2 className="text-2xl font-bold">Ready to get started?</h2>
      <p className="text-muted-foreground">Join thousands of users already using our platform.</p>
      <div className="flex gap-3 justify-center">
        <Button size="lg" rightIcon={<ChevronRight className="h-4 w-4" />}>
          Start Free Trial
        </Button>
        <Button size="lg" variant="outline">
          Learn More
        </Button>
      </div>
    </div>
  ),
};

export const SubmitButton: Story = {
  render: () => {
    // Simulated form submission
    return (
      <form
        onSubmit={(e) => {
          e.preventDefault();
          alert('Form submitted!');
        }}
        className="space-y-4 max-w-sm"
      >
        <div>
          <label className="text-sm font-medium">Email</label>
          <input
            type="email"
            placeholder="you@example.com"
            className="w-full mt-1 px-3 py-2 border rounded-md"
          />
        </div>
        <Button type="submit" fullWidth>
          Subscribe
        </Button>
      </form>
    );
  },
};

// ===========================================================================
// Accessibility Stories
// ===========================================================================

export const WithAriaLabel: Story = {
  args: {
    'aria-label': 'Close dialog',
    size: 'icon',
    variant: 'ghost',
    children: <X className="h-4 w-4" />,
  },
};

export const KeyboardFocus: Story = {
  render: () => (
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground">
        Press Tab to navigate between buttons and observe focus styles
      </p>
      <div className="flex gap-2">
        <Button>First</Button>
        <Button variant="outline">Second</Button>
        <Button variant="secondary">Third</Button>
      </div>
    </div>
  ),
};
