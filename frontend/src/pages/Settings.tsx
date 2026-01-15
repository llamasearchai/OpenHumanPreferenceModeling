import React from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Save, X } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { settingsApi } from '@/lib/api-client';
import { Skeleton } from '@/components/ui/skeleton';
import { Alert, AlertDescription } from '@/components/ui/alert';
import type { SettingsRequest } from '@/types/api';
import { mockSettings } from '@/lib/mock-data';
import { ApiRequestError, extractErrorMessage } from '@/lib/errors';

export default function SettingsPage() {
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const {
    data: settings,
    isLoading,
    error,
  } = useQuery({
    queryKey: ['settings'],
    queryFn: async () => {
      const result = await settingsApi.get();
      if (!result.success) {
        throw ApiRequestError.fromResponse(result);
      }
      return result.data;
    },
  });

  const updateMutation = useMutation({
    mutationFn: async (data: SettingsRequest) => {
      const result = await settingsApi.update(data);
      if (!result.success) {
        throw ApiRequestError.fromResponse(result);
      }
      return result.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['settings'] });
      toast({
        title: 'Settings saved',
        description: 'Your settings have been successfully updated.',
      });
    },
    onError: (error: Error) => {
      toast({
        title: 'Error',
        description: extractErrorMessage(error),
        variant: 'destructive',
      });
    },
  });

  const [formData, setFormData] = React.useState<SettingsRequest>({});

  const resolvedSettings = settings ?? (error ? mockSettings : undefined);
  const usingMockData = !!error && !settings;

  React.useEffect(() => {
    if (resolvedSettings) {
      setFormData({
        company_name: resolvedSettings.company_name,
        company_phone: resolvedSettings.company_phone,
        address: resolvedSettings.address,
        city: resolvedSettings.city,
        state: resolvedSettings.state,
        zip_code: resolvedSettings.zip_code,
        domain: resolvedSettings.domain,
        allowed_file_types: resolvedSettings.allowed_file_types,
        site_direction: resolvedSettings.site_direction,
        footer_info: resolvedSettings.footer_info,
      });
    }
  }, [resolvedSettings]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    updateMutation.mutate(formData);
  };

  const handleCancel = () => {
    if (resolvedSettings) {
      setFormData({
        company_name: resolvedSettings.company_name,
        company_phone: resolvedSettings.company_phone,
        address: resolvedSettings.address,
        city: resolvedSettings.city,
        state: resolvedSettings.state,
        zip_code: resolvedSettings.zip_code,
        domain: resolvedSettings.domain,
        allowed_file_types: resolvedSettings.allowed_file_types,
        site_direction: resolvedSettings.site_direction,
        footer_info: resolvedSettings.footer_info,
      });
    }
  };

  if (isLoading) {
    return (
      <div className="space-y-6 pt-6 pb-12">
        <div>
          <Skeleton className="h-10 w-48 mb-2" />
          <Skeleton className="h-5 w-96" />
        </div>
        <Card>
          <CardHeader>
            <Skeleton className="h-6 w-48" />
            <Skeleton className="h-4 w-96" />
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <Skeleton className="h-10 w-full" />
              <Skeleton className="h-10 w-full" />
              <Skeleton className="h-24 w-full" />
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-6 pt-6 pb-12">
      {usingMockData && (
        <Alert variant="warning">
          <AlertDescription>
            Settings could not be loaded: {extractErrorMessage(error, 'Unknown error')}.
            Showing mock data for validation.
          </AlertDescription>
        </Alert>
      )}
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Settings</h1>
        <p className="text-muted-foreground">
          Manage your company and site preferences
        </p>
      </div>

      <form onSubmit={handleSubmit}>
        <div className="grid gap-6">
          <Card>
            <CardHeader>
              <CardTitle>General Information</CardTitle>
              <CardDescription>
                Update your company details and site configuration.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid gap-4 md:grid-cols-2">
                <div className="space-y-2">
                  <Label htmlFor="companyName">Company Name</Label>
                  <Input
                    id="companyName"
                    value={formData.company_name || ''}
                    onChange={(e) => setFormData({ ...formData, company_name: e.target.value })}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="companyPhone">Phone</Label>
                  <Input
                    id="companyPhone"
                    value={formData.company_phone || ''}
                    onChange={(e) => setFormData({ ...formData, company_phone: e.target.value })}
                  />
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="address">Address</Label>
                <Input
                  id="address"
                  value={formData.address || ''}
                  onChange={(e) => setFormData({ ...formData, address: e.target.value })}
                />
              </div>

              <div className="grid gap-4 md:grid-cols-3">
                <div className="space-y-2">
                  <Label htmlFor="city">City</Label>
                  <Input
                    id="city"
                    value={formData.city || ''}
                    onChange={(e) => setFormData({ ...formData, city: e.target.value })}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="state">State</Label>
                  <Input
                    id="state"
                    value={formData.state || ''}
                    onChange={(e) => setFormData({ ...formData, state: e.target.value })}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="zip">Zip/Postal Code</Label>
                  <Input
                    id="zip"
                    value={formData.zip_code || ''}
                    onChange={(e) => setFormData({ ...formData, zip_code: e.target.value })}
                  />
                </div>
              </div>

              <div className="grid gap-4 md:grid-cols-2">
                <div className="space-y-2">
                  <Label htmlFor="domain">Domain</Label>
                  <Input
                    id="domain"
                    value={formData.domain || ''}
                    onChange={(e) => setFormData({ ...formData, domain: e.target.value })}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="files">Allowed File Types</Label>
                  <Input
                    id="files"
                    value={formData.allowed_file_types || ''}
                    onChange={(e) => setFormData({ ...formData, allowed_file_types: e.target.value })}
                  />
                </div>
              </div>

              <div className="space-y-3">
                <Label>Site Direction</Label>
                <RadioGroup
                  value={formData.site_direction || 'ltr'}
                  onValueChange={(value) => setFormData({ ...formData, site_direction: value as 'ltr' | 'rtl' })}
                  className="flex gap-4"
                >
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="ltr" id="ltr" />
                    <Label htmlFor="ltr">LTR (Left to Right)</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="rtl" id="rtl" />
                    <Label htmlFor="rtl">RTL (Right to Left)</Label>
                  </div>
                </RadioGroup>
              </div>

              <div className="space-y-2">
                <Label htmlFor="info">Footer Information (HTML/PDF)</Label>
                <Textarea
                  id="info"
                  className="min-h-[100px] font-mono text-xs"
                  value={formData.footer_info || ''}
                  onChange={(e) => setFormData({ ...formData, footer_info: e.target.value })}
                />
                <p className="text-[10px] text-muted-foreground">
                  Supported variables: &#123;company_name&#125;, &#123;address&#125;, &#123;city&#125;...
                </p>
              </div>

              <div className="flex justify-end gap-2">
                <Button type="button" variant="outline" onClick={handleCancel}>
                  <X className="mr-2 h-4 w-4" />
                  Cancel
                </Button>
                <Button type="submit" disabled={updateMutation.isPending}>
                  <Save className="mr-2 h-4 w-4" />
                  {updateMutation.isPending ? 'Saving...' : 'Save Changes'}
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </form>
    </div>
  );
}
