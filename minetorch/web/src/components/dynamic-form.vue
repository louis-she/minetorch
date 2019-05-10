<template>
  <div class="dynamic-form">
    <el-form ref="form" :model="formData" label-width="120px">
      <component v-for="field in schema" :is="field.type + '-field'" :key="field.label"
                 v-bind="field" :form-data="formData" />
      <el-form-item>
        <el-button type="primary" @click="onSubmit">Create</el-button>
        <el-button @click="$emit('dynamic-form:cancel')">Cancel</el-button>
      </el-form-item>
    </el-form>
  </div>
</template>
<script>
import string from './form-fields/string'
import select from './form-fields/select'
import boolean from './form-fields/boolean'

export default {
  components: {
    'string-field': string,
    'select-field': select,
    'boolean-field': boolean
  },
  props: {
    schema: {
      type: Array,
      default: function () {
        return [
        // {label: 'string field', type: 'string', help: 'this is a string', name: 'string_field'},
        // {label: 'select field', type: 'select', help: 'this is a select', name: 'select_field', options: [{label: 1, value: 1}, {label: 2, value: 2}]},
        // {label: 'multiselect field', type: 'select', multiple: 'true', help: 'this is a select', name: 'multi_select_field', options: [{label: 1, value: 1}, {label: 2, value: 2}]},
        // {label: 'boolean field', type: 'boolean', help: 'this is a boolean', name: 'boolean_field'}
        ]
      }
    },
    submitUrl: {
      type: String,
      default: ''
    },
    extraData: {
      type: Object,
      default: function() {
        return {}
      }
    }
  },
  data () {
    return {
      formData: {}
    }
  },
  methods: {
    async onSubmit() {
      const response = await this.ajax.post(this.submitUrl, Object.assign(this.extraData, this.formData))
    }
  }
}
</script>
<style lang="scss">
.el-input {
  width: 300px;
}
</style>
